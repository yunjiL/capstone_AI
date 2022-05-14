from dataclasses import dataclass
import os

class NUM_class:
    class_A: int = 0;
    class_B: int = 0;
    class_C: int = 0;

class Defect_info:
    Img_loc: str;
    defect: str;
    Class: chr;

class pipe_info:
    type: str;
    length: float;
    diameter: float;

pipe_cost = 0;

NUM_total_img = 0;

NUM_complete_pipe = 0;

def count_gauge(NUM_total_img, NUM_complete_pipe):
    calc_progress = NUM_complete_pipe % NUM_total_img;
    
    return calc_progress;

calc_progress = count_gauge(NUM_total_img, NUM_complete_pipe);

def calculate_risk(NUM_class):
    Defect_sum = 0;
    
    if NUM_class.class_A != 0:
        return 1;
    Defect_sum += NUM_class.class_B * 20;
    Defect_sum += NUM_class.class_C * 5;
    
    if Defect_sum >= 100:
        return 1;
    else:
        return 0;

threshold = calculate_risk(NUM_class);

def calculate_cost(pipe_info):
    pipe_cost = 0.0;
    
    if pipe_info.type == "CONC":
        pipe_cost=  pipe_info.length * pipe_info.diameter * 512740;
    elif pipe_info.type == "PLA":
        pipe_cost=  pipe_info.length * pipe_info.diameter * 707247;
    #더 추가
    return pipe_cost;

pipe_cost = calculate_cost(pipe_info);

