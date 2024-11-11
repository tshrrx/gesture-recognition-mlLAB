import numpy as np


def is_thumb_up(landmarks):
    return landmarks[4][1] < landmarks[3][1] and landmarks[4][1] < landmarks[5][1]

def is_fist(landmarks):
    return all(landmarks[i][1] < landmarks[i-2][1] for i in range(8, 20, 4))

def is_peace_sign(landmarks):
    return (landmarks[8][1] < landmarks[6][1] and  
            landmarks[12][1] < landmarks[10][1] and 
            landmarks[16][1] > landmarks[14][1] and 
            landmarks[20][1] > landmarks[18][1])    

def is_open_hand(landmarks):
    return (landmarks[4][1] < landmarks[3][1] and   
            landmarks[8][1] < landmarks[7][1] and   
            landmarks[12][1] < landmarks[11][1] and  
            landmarks[16][1] < landmarks[15][1] and  
            landmarks[20][1] < landmarks[19][1])     

def is_waving(landmarks):
    return landmarks[4][1] < landmarks[2][1] and landmarks[8][1] < landmarks[6][1]

def is_call_me(landmarks):
    return (landmarks[4][1] < landmarks[3][1] and  
            landmarks[20][1] < landmarks[19][1] and 
            landmarks[8][1] > landmarks[7][1] and   
            landmarks[12][1] > landmarks[11][1] and  
            landmarks[16][1] > landmarks[15][1])    
