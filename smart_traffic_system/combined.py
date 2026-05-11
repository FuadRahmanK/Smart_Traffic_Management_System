from flask import Flask ,render_template ,Response ,request ,redirect ,url_for ,jsonify
from ultralytics import YOLO
import cv2
import os
import queue
import threading
import time

app =Flask (__name__ )

UPLOAD_FOLDER ="static/uploads"
os .makedirs (UPLOAD_FOLDER ,exist_ok =True )
app .config ["UPLOAD_FOLDER"]=UPLOAD_FOLDER

vehicle_model =YOLO ("yolov8n.pt")
ambulance_model =YOLO ("best.pt") 

INFER_IMGSZ =416

import numpy as np
_warmup_frame =np .zeros ((INFER_IMGSZ ,INFER_IMGSZ ,3 ),dtype =np .uint8 )
vehicle_model (_warmup_frame ,imgsz =INFER_IMGSZ ,verbose =False )
ambulance_model (_warmup_frame ,imgsz =INFER_IMGSZ ,verbose =False )
del _warmup_frame

_ambulance_infer_sem =threading .Semaphore (2 )

_reset_event =threading .Event ()

MIN_GREEN =10
MAX_GREEN =60

AMBULANCE_GREEN_TIME =15
AMBULANCE_CHECK_DELAY =10
INITIAL_COMPUTE_TIME =5

CONF_THRESHOLD =0.85
AMBULANCE_CONFIRM_FRAMES =3
AMBULANCE_MEMORY_FRAMES =40

AMBULANCE_INFER_EVERY_N =5
VEHICLE_INFER_EVERY_N =3

YELLOW_TIME =5

TARGET_FPS =30
FRAME_DELAY =1.0 /TARGET_FPS

JPEG_QUALITY =[int (cv2 .IMWRITE_JPEG_QUALITY ),80 ]
ENCODE_SIZE =(640 ,480 )

STATE_CACHE_EVERY =6

CLEARANCE_TIME ={
"motorcycle":2 ,
"car":3 ,
"bus":4.5 ,
"truck":5 ,
}

traffic_lock =threading .Lock ()

caps ={}
stream_tokens ={i :0 for i in range (1 ,5 )}

display_queue ={i :queue .Queue (maxsize =2 )for i in range (1 ,5 )}
vehicle_queue ={i :queue .Queue (maxsize =1 )for i in range (1 ,5 )}
ambulance_queue ={i :queue .Queue (maxsize =1 )for i in range (1 ,5 )}

latest_frames ={}
frame_locks ={i :threading .Lock ()for i in range (1 ,5 )}

overlay_locks ={i :threading .Lock ()for i in range (1 ,5 )}
lane_annotations ={i :[]for i in range (1 ,5 )}
vehicle_annotations ={i :[]for i in range (1 ,5 )}

controller_thread =None

active_lane =1
next_lane =None
signal_state ="GREEN"
lane_start_time =time .time ()
lane_green_time =MIN_GREEN

lane_times ={i :0 for i in range (1 ,5 )}
waiting_score ={i :0 for i in range (1 ,5 )}

compute_active =False

ambulance_counter ={i :0 for i in range (1 ,5 )}
ambulance_memory ={i :0 for i in range (1 ,5 )}
ambulance_present ={i :False for i in range (1 ,5 )}

latest_vehicle_count ={i :0 for i in range (1 ,5 )}
latest_ambulance_detected ={i :False for i in range (1 ,5 )}

emergency_mode =False
emergency_lane_id =None
emergency_check_resume_time =0.0

def _pick_best_lane ():
    max_score =max (waiting_score .values ())
    candidates =[l for l ,v in waiting_score .items ()if v ==max_score ]
    if len (candidates )==1 :
        return candidates [0 ]
    return max (candidates ,key =lambda l :lane_times [l ])

def _trigger_emergency (lane_id ):
    global emergency_mode ,emergency_lane_id ,emergency_check_resume_time
    global active_lane ,next_lane ,signal_state ,lane_green_time ,lane_start_time
    global compute_active

    emergency_mode =True
    emergency_lane_id =lane_id
    active_lane =lane_id
    next_lane =None
    signal_state ="GREEN"
    lane_green_time =AMBULANCE_GREEN_TIME
    lane_start_time =time .time ()
    emergency_check_resume_time =time .time ()+AMBULANCE_CHECK_DELAY
    waiting_score [lane_id ]=0
    compute_active =False

    for i in range (1 ,5 ):
        _drain (vehicle_queue [i ])
        if i !=lane_id :
            _drain (ambulance_queue [i ])

def _clear_emergency (emg_lane ):
    global emergency_mode ,emergency_lane_id ,emergency_check_resume_time

    emergency_mode =False
    emergency_lane_id =None
    emergency_check_resume_time =0.0

    for i in range (1 ,5 ):
        ambulance_counter [i ]=0
        ambulance_memory [i ]=0
        ambulance_present [i ]=False
        if i !=emg_lane :
            waiting_score [i ]+=1
    waiting_score [emg_lane ]=0

def _drain (q ):
    while True :
        try :q .get_nowait ()
        except queue .Empty :break

@app .route ("/")
def upload_page ():
    return render_template ("up.html")

@app .route ("/upload",methods =["POST"])
def upload ():
    global controller_thread ,compute_active

    for i in range (1 ,5 ):
        file =request .files .get (f"lane{i}")
        if not (file and file .filename ):
            continue

        path =os .path .join (app .config ["UPLOAD_FOLDER"],file .filename )
        file .save (path )
        cap =cv2 .VideoCapture (path )

        with traffic_lock :
            old =caps .get (i )
            if old is not None :
                old .release ()
            caps [i ]=cap
            stream_tokens [i ]+=1
            token =stream_tokens [i ]

        for q in (display_queue [i ],vehicle_queue [i ],ambulance_queue [i ]):
            _drain (q )

        threading .Thread (target =_capture_thread ,args =(i ,token ),daemon =True ).start ()
        threading .Thread (target =_vehicle_thread ,args =(i ,token ),daemon =True ).start ()
        threading .Thread (target =_ambulance_thread ,args =(i ,token ),daemon =True ).start ()
        threading .Thread (target =_encode_thread ,args =(i ,token ),daemon =True ).start ()

    with traffic_lock :
        compute_active =True

    _reset_event .clear ()

    if controller_thread is None or not controller_thread .is_alive ():
        controller_thread =threading .Thread (target =_traffic_controller ,daemon =True )
        controller_thread .start ()

    return redirect (url_for ("dashboard"))

@app .route ("/dashboard")
def dashboard ():
    return render_template ("dashboard.html")

@app .route ("/signal_status")
def signal_status ():
    with traffic_lock :
        elapsed =time .time ()-lane_start_time
        remaining =max (0 ,lane_green_time -elapsed )
        return jsonify ({
        "active_lane":active_lane ,
        "next_lane":next_lane ,
        "signal_state":signal_state ,
        "remaining":int (remaining ),
        "emergency_lane":emergency_lane_id ,
        "vehicle_counts":dict (latest_vehicle_count ),
        "ambulance_detected":dict (latest_ambulance_detected ),
        })

@app .route ("/reset",methods =["POST"])
def reset ():
    global controller_thread ,compute_active
    global active_lane ,next_lane ,signal_state ,lane_green_time ,lane_start_time
    global emergency_mode ,emergency_lane_id ,emergency_check_resume_time

    _reset_event .set ()

    with traffic_lock :

        for i in range (1 ,5 ):
            stream_tokens [i ]+=1

        for i in range (1 ,5 ):
            cap =caps .pop (i ,None )
            if cap is not None :
                cap .release ()
            _drain (display_queue [i ])
            _drain (vehicle_queue [i ])
            _drain (ambulance_queue [i ])

        for i in range (1 ,5 ):
            lane_times [i ]=0
            waiting_score [i ]=0
            ambulance_counter [i ]=0
            ambulance_memory [i ]=0
            ambulance_present [i ]=False

        emergency_mode =False
        emergency_lane_id =None
        emergency_check_resume_time =0.0
        active_lane =1
        next_lane =None
        signal_state ="GREEN"
        lane_green_time =MIN_GREEN
        lane_start_time =time .time ()
        compute_active =False
        controller_thread =None

    for i in range (1 ,5 ):
        with overlay_locks [i ]:
            lane_annotations [i ]=[]
            vehicle_annotations [i ]=[]

    for i in range (1 ,5 ):
        with frame_locks [i ]:
            latest_frames .pop (i ,None )

    print ("[SYSTEM] Reset complete — ready for new upload.")
    return ("",204 )

def _traffic_controller ():
    global active_lane ,next_lane ,signal_state
    global lane_start_time ,lane_green_time
    global compute_active

    if _reset_event .wait (timeout =INITIAL_COMPUTE_TIME ):
        return
    with traffic_lock :
        compute_active =False

    while True :

        if _reset_event .is_set ():
            return

        with traffic_lock :
            if emergency_mode :
                if time .time ()>=lane_start_time +lane_green_time :
                    _clear_emergency (emergency_lane_id )

                else :

                    pass

        with traffic_lock :
            still_emergency =emergency_mode

        if still_emergency :
            _reset_event .wait (timeout =0.1 )
            continue

        if _reset_event .is_set ():
            return

        with traffic_lock :
            chosen =_pick_best_lane ()
            next_lane =chosen
            signal_state ="YELLOW"
            compute_active =True
            for i in range (1 ,5 ):
                lane_times [i ]=0
                _drain (vehicle_queue [i ])

        if _reset_event .wait (timeout =YELLOW_TIME ):
            return

        if _reset_event .is_set ():
            return

        with traffic_lock :
            if emergency_mode :
                compute_active =False
                for i in range (1 ,5 ):
                    _drain (vehicle_queue [i ])
                continue

        with traffic_lock :
            compute_active =False
            active_lane =next_lane
            next_lane =None
            signal_state ="GREEN"
            lane_green_time =max (MIN_GREEN ,min (lane_times [active_lane ],MAX_GREEN ))
            lane_start_time =time .time ()

            for lane in waiting_score :
                if lane ==active_lane :
                    waiting_score [lane ]=0
                else :
                    waiting_score [lane ]+=1

        deadline =time .time ()+lane_green_time
        while time .time ()<deadline :
            if _reset_event .is_set ():
                return
            with traffic_lock :
                if emergency_mode :
                    break
            _reset_event .wait (timeout =0.1 )

def _capture_thread (lane_id ,stream_token ):
    frame_count =0

    cached_cap =None
    cached_in_emg =False
    cached_is_emg_lane =False
    cached_resume_time =0.0
    cached_compute =False

    while True :

        if frame_count %STATE_CACHE_EVERY ==0 :
            with traffic_lock :
                if stream_tokens [lane_id ]!=stream_token :
                    return
                cached_cap =caps .get (lane_id )
                cached_in_emg =emergency_mode
                cached_is_emg_lane =(emergency_lane_id ==lane_id )
                cached_resume_time =emergency_check_resume_time
                cached_compute =compute_active

        if cached_cap is None :
            time .sleep (0.05 )
            continue

        ret ,frame =cached_cap .read ()
        if not ret :
            cached_cap .set (cv2 .CAP_PROP_POS_FRAMES ,0 )
            continue

        frame =cv2 .resize (frame ,(640 ,480 ))
        frame_count +=1

        try :display_queue [lane_id ].put_nowait ((frame .copy (),frame_count ))
        except queue .Full :pass

        now =time .time ()

        if cached_compute and not cached_in_emg :
            if frame_count %VEHICLE_INFER_EVERY_N ==0 :
                try :vehicle_queue [lane_id ].put_nowait (frame .copy ())
                except queue .Full :pass

        if cached_in_emg :
            if cached_is_emg_lane and now >=cached_resume_time :
                if frame_count %AMBULANCE_INFER_EVERY_N ==0 :
                    try :ambulance_queue [lane_id ].put_nowait (frame .copy ())
                    except queue .Full :pass
        else :
            if frame_count %AMBULANCE_INFER_EVERY_N ==0 :
                try :ambulance_queue [lane_id ].put_nowait (frame .copy ())
                except queue .Full :pass

        time .sleep (FRAME_DELAY )

def _vehicle_thread (lane_id ,stream_token ):

    token_check =0

    while True :

        token_check +=1
        if token_check %STATE_CACHE_EVERY ==0 :
            with traffic_lock :
                if stream_tokens [lane_id ]!=stream_token :
                    return

        try :
            frame =vehicle_queue [lane_id ].get (timeout =1.0 )
        except queue .Empty :
            continue

        frame_height =frame .shape [0 ]
        results =vehicle_model (frame ,conf =0.4 ,imgsz =INFER_IMGSZ )[0 ]
        raw_time =0.0

        vehicle_counts ={}
        contributions =[]

        for box in results .boxes :
            cls =int (box .cls )
            label =vehicle_model .names [cls ]
            if label not in CLEARANCE_TIME :
                continue
            x1 ,y1 ,x2 ,y2 =map (int ,box .xyxy [0 ])
            y_center =(y1 +y2 )/2
            distance_factor =0.5 +(1.0 -(y_center /frame_height ))
            contrib =CLEARANCE_TIME [label ]*distance_factor
            raw_time +=contrib

            vehicle_counts [label ]=vehicle_counts .get (label ,0 )+1
            contributions .append ((label ,CLEARANCE_TIME [label ],
            distance_factor ,contrib ))

        if contributions :
            green_time =max (MIN_GREEN ,min (raw_time ,MAX_GREEN ))
            lines =[f"[Lane {lane_id}] Clearance calculation:"]
            for label ,base_ct ,dist ,contrib in contributions :
                lines .append (
                f"  {label:<12} base={base_ct:.1f}s  "
                f"dist_factor={dist:.2f}  contrib={contrib:.2f}s"
                )
            lines .append (
            f"  {'TOTAL':<12} raw={raw_time:.2f}s  "
            f"green_time={green_time:.0f}s "
            f"(clamped to [{MIN_GREEN}, {MAX_GREEN}])"
            )
            print ("\n".join (lines ))

        vehicle_boxes =[]
        for box in results .boxes :
            cls =int (box .cls )
            label =vehicle_model .names [cls ]
            if label not in CLEARANCE_TIME :
                continue
            x1 ,y1 ,x2 ,y2 =map (int ,box .xyxy [0 ])
            vehicle_boxes .append ((int (x1 ),int (y1 ),int (x2 ),int (y2 ),label ))

        with traffic_lock :

            if raw_time >lane_times [lane_id ]:
                lane_times [lane_id ]=raw_time

            latest_vehicle_count [lane_id ]=sum (vehicle_counts .values ())

        with overlay_locks [lane_id ]:
            vehicle_annotations [lane_id ]=vehicle_boxes

def _ambulance_thread (lane_id ,stream_token ):

    while True :
        with traffic_lock :
            if stream_tokens [lane_id ]!=stream_token :
                return

        try :
            frame =ambulance_queue [lane_id ].get (timeout =1.0 )
        except queue .Empty :
            continue

        with traffic_lock :
            in_emg =emergency_mode
            emg_lane =emergency_lane_id

        if in_emg and emg_lane !=lane_id :
            continue

        _run_ambulance_inference (frame ,lane_id ,in_emg )

def _run_ambulance_inference (frame ,lane_id ,in_emergency ):

    with _ambulance_infer_sem :
        results =ambulance_model (frame ,imgsz =INFER_IMGSZ )[0 ]

    detected_this_frame =any (
    float (b .conf [0 ])>=CONF_THRESHOLD for b in results .boxes
    )
    annotations =[
    (int (b .xyxy [0 ][0 ]),int (b .xyxy [0 ][1 ]),
    int (b .xyxy [0 ][2 ]),int (b .xyxy [0 ][3 ]),"AMBULANCE")
    for b in results .boxes if float (b .conf [0 ])>=CONF_THRESHOLD
    ]

    with overlay_locks [lane_id ]:
        lane_annotations [lane_id ]=annotations

    should_clear_all_overlays =False

    with traffic_lock :

        if in_emergency :

            if detected_this_frame :
                ambulance_counter [lane_id ]=AMBULANCE_CONFIRM_FRAMES
                ambulance_memory [lane_id ]=AMBULANCE_MEMORY_FRAMES
                _trigger_emergency (lane_id )
            else :
                _clear_emergency (lane_id )
                should_clear_all_overlays =True

        else :

            if detected_this_frame :
                ambulance_counter [lane_id ]+=1
            else :
                ambulance_counter [lane_id ]=0

            if ambulance_counter [lane_id ]>=AMBULANCE_CONFIRM_FRAMES :
                ambulance_memory [lane_id ]=AMBULANCE_MEMORY_FRAMES

            if ambulance_memory [lane_id ]>0 :
                ambulance_detected =True
                ambulance_present [lane_id ]=True
                ambulance_memory [lane_id ]-=1
                if ambulance_memory [lane_id ]==0 :
                    ambulance_present [lane_id ]=False
            else :
                ambulance_detected =False
                ambulance_present [lane_id ]=False

            latest_ambulance_detected [lane_id ]=ambulance_detected

            if ambulance_detected :
                print (f"[Lane {lane_id}] AMBULANCE DETECTED")
                _trigger_emergency (lane_id )

    if should_clear_all_overlays :
        for i in range (1 ,5 ):
            with overlay_locks [i ]:
                lane_annotations [i ]=[]
                vehicle_annotations [i ]=[]

def _encode_thread (lane_id ,stream_token ):

    token_check =0
    cached_signal_state ="GREEN"

    while True :

        token_check +=1
        if token_check %STATE_CACHE_EVERY ==0 :
            with traffic_lock :
                if stream_tokens [lane_id ]!=stream_token :
                    return
                cached_signal_state =signal_state

        try :
            frame ,_ =display_queue [lane_id ].get (timeout =1.0 )
        except queue .Empty :
            continue

        with overlay_locks [lane_id ]:
            ambulance_boxes =list (lane_annotations [lane_id ])
        for x1 ,y1 ,x2 ,y2 ,label in ambulance_boxes :
            cv2 .rectangle (frame ,(x1 ,y1 ),(x2 ,y2 ),(0 ,0 ,255 ),2 )
            cv2 .putText (frame ,label ,(x1 ,y1 -8 ),
            cv2 .FONT_HERSHEY_SIMPLEX ,0.5 ,(0 ,0 ,255 ),2 )

        if cached_signal_state =="YELLOW":
            with overlay_locks [lane_id ]:
                vehicle_boxes =list (vehicle_annotations [lane_id ])
            for x1 ,y1 ,x2 ,y2 ,label in vehicle_boxes :
                cv2 .rectangle (frame ,(x1 ,y1 ),(x2 ,y2 ),(0 ,255 ,255 ),2 )
                cv2 .putText (frame ,label ,(x1 ,y1 -8 ),
                cv2 .FONT_HERSHEY_SIMPLEX ,0.5 ,(0 ,255 ,255 ),1 )
        else :

            with overlay_locks [lane_id ]:
                vehicle_annotations [lane_id ]=[]

        cv2 .putText (frame ,f"Road {lane_id}",(8 ,24 ),
        cv2 .FONT_HERSHEY_SIMPLEX ,0.7 ,(255 ,255 ,255 ),2 )

        ret ,buffer =cv2 .imencode (".jpg",frame ,JPEG_QUALITY )
        if ret :
            with frame_locks [lane_id ]:
                latest_frames [lane_id ]=buffer

@app .route ("/video_feed/<int:lane_id>")
def video_feed (lane_id ):

    def generate ():
        last_frame =None
        while True :
            with frame_locks [lane_id ]:
                frame =latest_frames .get (lane_id )

            if frame is None :
                time .sleep (0.02 )
                continue
            if frame is last_frame :
                time .sleep (0.01 )
                continue

            last_frame =frame
            yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +frame .tobytes ()
            +b"\r\n"
            )

    return Response (
    generate (),
    mimetype ="multipart/x-mixed-replace; boundary=frame",
    )

if __name__ =="__main__":
    app .run (debug =True ,threaded =True ,use_reloader =False )