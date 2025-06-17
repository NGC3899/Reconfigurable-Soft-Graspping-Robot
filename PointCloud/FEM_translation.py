# -*- coding: utf-8 -*-
# Version: 5.17 (Use Hardcoded Target RP Integer Key)

# =============================================================================
# Import necessary modules
# =============================================================================
import os
import sys
import datetime
import traceback
import math

# --- 初始化 Debug Excel 日志 ---
target_directory = r'C:\Users\admin\Desktop\FEM'
debug_excel_file_name = 'Debug_Log.xlsx'
debug_excel_file = os.path.join(target_directory, debug_excel_file_name)
initial_log_success = False
wb_debug = None
ws_debug = None
log_init_dir_msg = None

# --- 确保目录存在 (V5.17 严格结构) ---
print "DEBUG: Checking target directory: {}".format(target_directory) # 独立行
# if 块开始
if not os.path.exists(target_directory):
    # try 块开始
    try:
        print "DEBUG: Target directory does not exist. Attempting creation..." # 独立行
        os.makedirs(target_directory) # 独立行
        log_init_dir_msg = "INFO: Created target directory: {}".format(target_directory) # 独立行
        print log_init_dir_msg # 独立行
    # except OSError 块开始
    except OSError as e:
        print "FATAL ERROR: Cannot create target directory for logging: {}. Error: {}".format(target_directory, e) # 独立行
        sys.exit("Script aborted: cannot create log directory.") # 独立行
    # except Exception 块开始
    except Exception as e_create:
         print "FATAL ERROR: Unexpected error creating directory: {}. Error: {}".format(target_directory, e_create) # 独立行
         sys.exit("Script aborted: unexpected directory creation error.") # 独立行
# else 块开始
else:
    log_init_dir_msg = "DEBUG: Target directory already exists: {}".format(target_directory) # 独立行
    print log_init_dir_msg # 独立行

# --- 尝试导入 openpyxl 并初始化 Debug Excel (V5.17 严格结构) ---
try:
    print "DEBUG: Importing openpyxl..." # 独立行
    import openpyxl # 独立行
    from openpyxl import Workbook # 独立行
    print "DEBUG: Initializing Debug Workbook..." # 独立行
    wb_debug = Workbook() # 独立行
    ws_debug = wb_debug.active # 独立行
    ws_debug.title = 'DebugLog' # 独立行
    print "DEBUG: Appending headers to Debug Log..." # 独立行
    ws_debug.append(['Timestamp', 'Level', 'Message', 'Traceback']) # 独立行
    initial_log_success = True # 独立行
    print "DEBUG: Debug Excel log initialized." # 独立行
# except ImportError 块开始
except ImportError:
    print "FATAL ERROR: The 'openpyxl' library is required for logging but not found." # 独立行
    print "Please install it in your Abaqus Python environment." # 独立行
    print "Run: abaqus python -m pip install openpyxl" # 独立行
    sys.exit("Script aborted due to missing openpyxl for logging.") # 独立行
# except Exception 块开始
except Exception as e:
    print "FATAL ERROR: Failed to initialize Debug Excel Log ({}). Error: {}".format(debug_excel_file, e) # 独立行
    sys.exit("Script aborted: Cannot initialize debug log.") # 独立行

# --- 定义日志记录函数 ---
def log_to_excel(level, message, exc_obj=None):
    # if 块开始
    if not initial_log_success or ws_debug is None:
        print "LOG_FALLBACK: [{}] {}".format(level, message) # 独立行
        # if 块开始
        if exc_obj:
            print "LOG_FALLBACK_EXCEPTION: {}".format(exc_obj) # 独立行
        return # 独立行
    # try 块开始
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tb_str = "" # 独立行
        # if 块开始
        if exc_obj:
            # try 块开始
            try:
                tb_str = "".join(traceback.format_exception(type(exc_obj), exc_obj, exc_obj.__traceback__)) # 独立行
            # except 块开始
            except Exception as format_exc_err:
                try: # 尝试获取 repr
                    original_exc_repr = repr(exc_obj) # 独立行
                except Exception: # 获取 repr 也失败
                    original_exc_repr = "[Could not get repr() of original exception]" # 独立行
                # 记录包含类型和 repr 的回退信息
                tb_str = "Traceback formatting failed (Error: {}). Original Exception Type: <{}>, Original Exception Repr: {}".format(
                    format_exc_err, type(exc_obj).__name__, original_exc_repr
                ) # 多行赋值保持可读性
        ws_debug.append([str(timestamp), str(level), str(message), str(tb_str)]) # 独立行
    # except 块开始
    except Exception as log_e:
        print "ERROR writing to debug log: {}. Original message: [{}] {}".format(log_e, level, message) # 独立行

# --- 定义保存 Debug Excel 的函数 ---
def save_debug_log():
    # if 块开始
    if not initial_log_success or wb_debug is None:
         print "ERROR: Debug log was not initialized, cannot save." # 独立行
         return False # 独立行
    # try 块开始
    try:
        wb_debug.save(debug_excel_file) # 独立行
        print "Attempting to save debug log to {}".format(debug_excel_file) # 独立行
        return True # 独立行
    # except 块开始
    except Exception as save_e:
        print "ERROR: Failed to save debug log file '{}'. Error: {}".format(debug_excel_file, save_e) # 独立行
        return False # 独立行

# --- 开始记录日志 ---
# if 块开始
if initial_log_success and log_init_dir_msg:
    log_to_excel("INFO", log_init_dir_msg.replace("DEBUG: ", "Directory Info: ")) # 独立行
# elif 块开始
elif initial_log_success:
    log_to_excel("INFO", "Debug Excel log active.") # 独立行

log_to_excel("DEBUG", "Importing Abaqus modules...")
from abaqus import *
from abaqusConstants import *
import odbAccess
import job
import load
import step
import assembly # 确保导入 assembly

# 尝试导入 numpy (严格分行)
# try 块开始
try:
    import numpy as np
    log_to_excel("DEBUG", "Numpy imported successfully.")
# except ImportError 块开始
except ImportError as e:
    err_msg = "The 'numpy' library is required..." # 省略
    log_to_excel("ERROR", err_msg, exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted due to missing numpy.") # 独立行

log_to_excel("DEBUG", "Standard Python modules imported.")

# =============================================================================
# USER CONFIGURABLE PARAMETERS - MODIFY THESE VALUES
# =============================================================================
log_to_excel("DEBUG", "Setting user configurable parameters...")
cae_file_path = r'D:\softgripper3.cae'
model_name = 'Model-1'
pressure_load_names = [
    'pressure1', 'pressure2', 'pressure3', 'pressure4a', 'pressure4b', 'pressure4c',
    'pressure4d', 'pressure4e', 'pressure4f', 'pressure5a', 'pressure5b', 'pressure5c',
    'pressure5d', 'pressure5e'
]
num_pressure_iterations = 10
initial_pressure = 10000.0
pressure_increment = 1000.0
excel_file_name = 'Max_Contact_Stress_Results.xlsx'
output_excel_file = os.path.join(target_directory, excel_file_name)

# --- Rigid Body Position Control ---
rigid_cube_instance_name = 'cube-1' # 刚体实例名称
rigid_cube_tracking_node_label = 601 # 用于获取初始位置的节点标签
# <<< --- 修改：不再使用 Target RP 名称，改用其整数 Key --- >>>
# target_rp_name = 'RP_TARGET' # 不再需要名称参数
# !!! --- 请填入你在 CAE 中找到的 'RP_TARGET' 对应的整数 Key (例如 25 或 33) --- !!!
target_rp_key = 33 # <<< 在这里填入你确定的整数 Key！
# !!! --------------------------------------------------------------------- !!!

# --- 偏移量参数 ---
x_offset_start = -0.1
x_offset_end = 0.1
x_steps = 3
z_offset_start = -0.1
z_offset_end = 0.1
z_steps = 2

# --- 使用 Node Set 提取结果 ---
contact_node_set_name = 'NS_CONTACT_SLAVE' # <<< 结果提取节点集名称

step_name = 'Bending'
output_variables = ["CPRESS", "CSHEAR1", "CSHEAR2"]
# ... (参数日志) ...
log_to_excel("DEBUG", "User parameters set.")
log_to_excel("DEBUG", "Target output directory: {}".format(target_directory))
log_to_excel("DEBUG", "Output Result file path: {}".format(output_excel_file))
log_to_excel("DEBUG", "Debug Excel file path: {}".format(debug_excel_file))
log_to_excel("DEBUG", "Rigid cube instance: '{}'".format(rigid_cube_instance_name))
log_to_excel("DEBUG", "Tracking Node Label (for initial pos): {}".format(rigid_cube_tracking_node_label))
log_to_excel("DEBUG", "Target RP Key (for positioning): {}".format(target_rp_key)) # <<< 修改日志
log_to_excel("DEBUG", "X Offset Range: {} to {} in {} steps".format(x_offset_start, x_offset_end, x_steps))
log_to_excel("DEBUG", "Z Offset Range: {} to {} in {} steps".format(z_offset_start, z_offset_end, z_steps))
log_to_excel("DEBUG", "Pressure Range: {} iterations...".format(num_pressure_iterations))
log_to_excel("DEBUG", "Using Node Set for region: {}".format(contact_node_set_name))

# =============================================================================
# Script Initialization
# =============================================================================
log_to_excel("INFO", "Starting Abaqus parametric pressure study script...")

# --- Initialize RESULT Excel Workbook ---
log_to_excel("DEBUG", "Initializing Result Excel workbook...")
excel_headers = ["Pressure Iter", "Pressure", "X Iter", "X Offset", "Z Iter", "Z Offset"] + ["Max " + var for var in output_variables]
wb_result = None
ws_result = None
try:
    wb_result = Workbook() # 独立行
    ws_result = wb_result.active # 独立行
    ws_result.title = 'Max Contact Data' # 独立行
    log_to_excel("DEBUG", "Appending headers to Result Excel sheet...") # 独立行
    ws_result.append(excel_headers) # 独立行
    log_to_excel("DEBUG", "Result Excel headers written: {}".format(excel_headers)) # 独立行
except Exception as e:
    log_to_excel("ERROR", "Failed to initialize Result Excel workbook.", exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted: Failed to initialize result excel.") # 独立行


# --- Open the CAE file and access the model ---
log_to_excel("DEBUG", "Attempting to open CAE file: {}".format(cae_file_path))
myModel = None
myAssembly = None
cube_instance = None
initial_x0 = 0.0
initial_y0 = 0.0
initial_z0 = 0.0
tracking_node = None
target_rp = None # 目标 RP 对象
# <<< 严格检查并修正分行 >>>
try:
    openMdb(pathName=cae_file_path) # 独立行
    log_to_excel("INFO", "Successfully opened CAE file: {}".format(cae_file_path)) # 独立行
    myModel = mdb.models[model_name] # 独立行
    log_to_excel("INFO", "Accessed model: {}".format(model_name)) # 独立行
    myAssembly = myModel.rootAssembly # 独立行
    cube_instance = myAssembly.instances[rigid_cube_instance_name] # 独立行
    log_to_excel("INFO", "Accessed rigid cube instance: {}".format(rigid_cube_instance_name)) # 独立行

    # --- 获取追踪节点的初始坐标 ---
    # ... (查找追踪节点的逻辑不变) ...
    log_to_excel("DEBUG", "Accessing tracking node Label {} for initial position...".format(rigid_cube_tracking_node_label)) # 独立行
    node_found_flag = False # 独立行
    tracking_node = None # 独立行
    try: # 尝试索引
        tracking_node = cube_instance.nodes[rigid_cube_tracking_node_label - 1] # 独立行
        if tracking_node and tracking_node.label == rigid_cube_tracking_node_label: # 独立行
            node_found_flag = True # 独立行
        else: # 独立行
            tracking_node = None # 独立行
    except (KeyError, IndexError, TypeError, AttributeError): # except 块开始
         log_to_excel("WARNING","Direct access using label-1 index failed. Trying iteration...") # 独立行
         tracking_node = None # 独立行

    if not node_found_flag: # if 块开始 (尝试迭代)
        log_to_excel("DEBUG", "Trying iteration to find node with label {}...".format(rigid_cube_tracking_node_label)) # 独立行
        if hasattr(cube_instance, 'nodes') and cube_instance.nodes: # if 块开始
            for node in cube_instance.nodes: # for 块开始
                if node.label == rigid_cube_tracking_node_label: # if 块开始
                    tracking_node = node # 独立行
                    node_found_flag = True # 独立行
                    log_to_excel("DEBUG","Node found via iteration.") # 独立行
                    break # 独立行
        else: # else 块开始
             log_to_excel("WARNING", "Instance nodes attribute missing or empty.") # 独立行

        if not node_found_flag: # if 块开始
             log_to_excel("ERROR", "Node with label {} not found.".format(rigid_cube_tracking_node_label)) # 独立行
             raise KeyError("Tracking node label not found.") # 独立行

    log_to_excel("INFO", "Successfully accessed tracking node (Label {})".format(rigid_cube_tracking_node_label)) # 独立行
    initial_coords = tracking_node.coordinates # 独立行
    initial_x0 = initial_coords[0] # 独立行
    initial_y0 = initial_coords[1] # 独立行
    initial_z0 = initial_coords[2] # 独立行
    log_to_excel("INFO", "Initial tracking node coords: X={}, Y={}, Z={}".format(initial_x0, initial_y0, initial_z0)) # 独立行

    # --- <<< V5.17: 直接使用整数 Key 获取 Target RP 对象 >>> ---
    log_to_excel("DEBUG", "Attempting to access Target Reference Point using integer key: {}...".format(target_rp_key)) # 独立行
    target_rp = None # 初始化
    try: # try 块开始
        # 直接用 Key 访问 Assembly 的 RPs 库
        target_rp = myAssembly.referencePoints[target_rp_key] # <<< 使用整数 Key >>>
        log_to_excel("INFO", "Successfully accessed Target Reference Point object using key {}: {}".format(target_rp_key, target_rp)) # 独立行
        # 可选：尝试记录坐标以验证
        try: # try 块开始
             rp_coords_check = target_rp.pointOn[0] # 独立行
             log_to_excel("DEBUG", "Coordinates of RP with key {}: {}".format(target_rp_key, rp_coords_check)) # 独立行
        except Exception as coord_e: # except 块开始
             log_to_excel("WARNING", "Could not get coordinates for RP with key {}.".format(target_rp_key), exc_obj=coord_e) # 独立行

    except KeyError as e_rp_key: # except 块开始 (捕获 Key 找不到的错误)
        log_to_excel("ERROR", "Could not find Target Reference Point with integer key {} in assembly. Available keys: {}".format(target_rp_key, myAssembly.referencePoints.keys()), exc_obj=e_rp_key) # 独立行
        save_debug_log() # 独立行
        sys.exit("Script aborted: Invalid Target RP Key specified.") # 独立行
    except Exception as e_get_rp: # except 块开始 (捕获其他访问错误)
        log_to_excel("ERROR", "An unexpected error occurred accessing Target RP with key {}.".format(target_rp_key), exc_obj=e_get_rp) # 独立行
        save_debug_log() # 独立行
        sys.exit("Script aborted during Target RP access.") # 独立行
    # --- <<< 修改结束 --- >>>

except MdbError as e: # except 块开始
    log_to_excel("ERROR", "Failed to open CAE file '{}'. MdbError.".format(cae_file_path), exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted.") # 独立行
except KeyError as e: # except 块开始
    # ... (KeyError handling) ...
    error_message = str(e); key_error_source = "Unknown" # 独立行
    if model_name in error_message: key_error_source = "Model '{}'".format(model_name) # 独立行
    elif rigid_cube_instance_name in error_message: key_error_source = "Instance '{}'".format(rigid_cube_instance_name) # 独立行
    elif str(target_rp_key) in error_message: key_error_source = "Target RP Key {}".format(target_rp_key) # 独立行
    elif str(rigid_cube_tracking_node_label) in error_message: key_error_source = "Tracking Node Label '{}'".format(rigid_cube_tracking_node_label) # 独立行
    log_to_excel("ERROR", "KeyError accessing MDB object: {}. Details: {}".format(key_error_source, e), exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted.") # 独立行
except IndexError as e_idx: # except 块开始
     log_to_excel("ERROR", "Could not access node with Label {}...".format(rigid_cube_tracking_node_label), exc_obj=e_idx) # 独立行
     save_debug_log() # 独立行
     sys.exit("Script aborted.") # 独立行
except Exception as e: # except 块开始
    log_to_excel("ERROR", "Unexpected error opening CAE/accessing initial objects.", exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted.") # 独立行


# --- 更改当前工作目录到目标文件夹 ---
log_to_excel("DEBUG", "Changing current working directory to: {}".format(target_directory))
# <<< 严格检查并修正分行 >>>
try:
    os.chdir(target_directory) # 独立行
    log_to_excel("INFO", "Current working directory changed successfully.") # 独立行
except OSError as e:
    log_to_excel("ERROR", "Could not change working directory to '{}'...".format(target_directory), exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted due to chdir error.") # 独立行

# =============================================================================
# Main Simulation Loops (Triple Nested)
# =============================================================================
log_to_excel("INFO", "Entering main simulation loops...")
# ... (计算偏移增量不变) ...
x_offset_increment = (x_offset_end - x_offset_start) / (x_steps - 1) if x_steps > 1 else 0.0
z_offset_increment = (z_offset_end - z_offset_start) / (z_steps - 1) if z_steps > 1 else 0.0
log_to_excel("DEBUG", "X offset increment: {}".format(x_offset_increment))
log_to_excel("DEBUG", "Z offset increment: {}".format(z_offset_increment))

# --- 压力循环 ---
for i in range(num_pressure_iterations):
    # ... (循环和压力计算不变) ...
    pressure_iter_num = i + 1
    current_pressure = initial_pressure + i * pressure_increment
    log_to_excel("INFO", "--- Starting P Iter {}/{} (P={:.2f}) ---".format(pressure_iter_num, num_pressure_iterations, current_pressure))

    # --- 修改压力载荷 ---
    log_to_excel("DEBUG", "Updating pressure loads...")
    # <<< 严格检查并修正分行 >>>
    try:
        if myModel is None:
            raise ValueError("MDB Model unavailable.") # 独立行
        for load_name in pressure_load_names:
            load_object = myModel.loads[load_name] # 独立行
            load_object.setValues(magnitude=current_pressure) # 独立行
        log_to_excel("DEBUG", "Pressure loads updated.") # 独立行
    except KeyError as e:
        log_to_excel("ERROR", "Pressure load name '{}' not found...".format(e), exc_obj=e) # 独立行
        save_debug_log() # 独立行
        sys.exit("Script aborted.") # 独立行
    except Exception as e:
        log_to_excel("ERROR", "Failed to update pressure loads.", exc_obj=e) # 独立行
        save_debug_log() # 独立行
        sys.exit("Script aborted.") # 独立行

    # --- X 位置偏移循环 ---
    for j in range(x_steps):
        # ... (X 循环和坐标计算不变) ...
        x_iter_num = j + 1
        if x_steps == 1:
            current_x_offset = x_offset_start
        else:
            current_x_offset = x_offset_start + j * x_offset_increment
        x_target_abs = initial_x0 + current_x_offset
        log_to_excel("INFO", "----- Starting X Iter {}/{} (X_off={:.4f}) -----".format(x_iter_num, x_steps, current_x_offset))

        # --- Z 位置偏移循环 ---
        for k in range(z_steps):
            # ... (Z 循环和坐标计算不变) ...
            z_iter_num = k + 1
            if z_steps == 1:
                current_z_offset = z_offset_start
            else:
                current_z_offset = z_offset_start + k * z_offset_increment
            z_target_abs = initial_z0 + current_z_offset
            log_to_excel("INFO", "------- Starting Z Iter {}/{} (Z_off={:.4f}) -------".format(z_iter_num, z_steps, current_z_offset))

            # --- 修改目标 RP 坐标并重新生成 ---
            log_to_excel("DEBUG", "Preparing to reposition cube via Target RP (Key {})".format(target_rp_key)) # 独立行
            positioning_success = False # 独立行
            # <<< 严格检查并修正分行 >>>
            try:
                # 1. 确定目标坐标
                target_coords = (x_target_abs, initial_y0, z_target_abs) # 独立行
                log_to_excel("INFO", "Setting Target RP (Key {}) coordinates to: {}".format(target_rp_key, target_coords)) # 独立行

                # 2. 修改参考点坐标
                if target_rp is None: # 独立行
                    raise ValueError("Target RP object is None.") # 独立行

                log_to_excel("DEBUG", "Calling myAssembly.editReferencePoint...") # 独立行
                myAssembly.editReferencePoint(referencePoint=target_rp, point=target_coords) # 独立行
                log_to_excel("DEBUG", "Target RP coordinates edited.") # 独立行

                # 3. 重新生成装配体
                log_to_excel("INFO", "Regenerating assembly...") # 独立行
                myAssembly.regenerate() # 独立行
                log_to_excel("INFO", "Assembly regenerated.") # 独立行
                positioning_success = True # 独立行

            except AttributeError as e_attr: # except 块开始
                 log_to_excel("ERROR", "AttributeError during RP edit/regen.", exc_obj=e_attr) # 独立行
                 log_to_excel("WARNING", "Cannot reposition cube. Skipping.") # 独立行
                 continue # 独立行
            except ValueError as e_val: # except 块开始
                 log_to_excel("ERROR", "ValueError during RP edit/regen.", exc_obj=e_val) # 独立行
                 log_to_excel("WARNING", "Cannot reposition cube. Skipping.") # 独立行
                 continue # 独立行
            except Exception as move_e: # except 块开始
                log_to_excel("ERROR", "Failed to reposition cube instance '{}' via Target RP Key {}.".format(rigid_cube_instance_name, target_rp_key), exc_obj=move_e) # 独立行
                warning_message = "Cannot reposition cube. Skipping iteration. Reason: {}".format(move_e) # 独立行
                log_to_excel("WARNING", warning_message) # 独立行
                continue # 独立行

            # Check success flag
            if not positioning_success: # 独立行
                log_to_excel("WARNING", "Skipping Job/ODB due to failed repositioning.") # 独立行
                continue # 独立行


            # --- 创建并运行 Job ---
            job_name = 'P{}_X{}_Z{}'.format(pressure_iter_num, x_iter_num, z_iter_num)
            # ... (Job 创建、提交、等待逻辑不变, 确保 except 分行) ...
            log_to_excel("DEBUG", "Creating Job: {}".format(job_name)); myJob = None
            try:
                if myModel is None: raise ValueError("MDB Model unavailable...")
                myJob = mdb.Job(name=job_name, model=model_name, description='P{},X{:.2f},Z{:.2f}'.format(current_pressure,current_x_offset,current_z_offset), type=ANALYSIS, #...
                                resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, numDomains=1, numGPUs=0)
                log_to_excel("DEBUG", "Job object created."); log_to_excel("INFO", "Submitting job: {}".format(job_name))
                myJob.submit(consistencyChecking=OFF); log_to_excel("DEBUG", "Job submission issued.")
                log_to_excel("INFO", "Waiting for job: {}...".format(job_name)); myJob.waitForCompletion()
                log_to_excel("INFO", "Job {} completed.".format(job_name))
            except AbaqusException as e:
                log_to_excel("ERROR", "Abaqus job failed for {}.".format(job_name), exc_obj=e)
                log_to_excel("WARNING", "Skipping results...")
                continue
            except Exception as e:
                log_to_excel("ERROR", "Unexpected job error for {}.".format(job_name), exc_obj=e)
                log_to_excel("WARNING", "Skipping results...")
                continue


            # --- Post-processing: Extract results from ODB ---
            odb_path = job_name + '.odb'
            # ... (ODB 打开、获取 Frame、提取最大值的逻辑不变) ...
            # --- <<< ODB NodeSet 访问方式已在 V5.8 中修正 >>> ---
            log_to_excel("INFO", "Starting ODB post-processing for: {}".format(odb_path)); odb = None
            max_values_for_frame = {var: 0.0 for var in output_variables}; log_to_excel("DEBUG", "Initialized max_values...")
            try: # ODB Main Try
                log_to_excel("DEBUG", "Opening ODB: {}".format(odb_path)) # 独立行
                if not os.path.exists(odb_path): err_msg = "... ODB not found ..."; log_to_excel("ERROR", err_msg); raise odbAccess.OdbError(err_msg)
                odb = odbAccess.openOdb(path=odb_path, readOnly=True); log_to_excel("INFO", "Opened ODB: {}".format(odb_path)) # 独立行
                # --- 获取 ODB NodeSet 对象 ---
                node_set_region = None; log_to_excel("DEBUG", "Accessing NodeSet: '{}' in ODB...".format(contact_node_set_name)) # 独立行
                odb_node_sets = odb.rootAssembly.nodeSets; found_odb_node_set = None; available_odb_node_sets = {} # 独立行
                if not odb_node_sets or len(odb_node_sets.keys()) == 0: log_to_excel("WARNING", "ODB has no nodeSets.") # 独立行
                else:
                    log_to_excel("DEBUG", "Available ODB NodeSet Keys: {}".format(odb_node_sets.keys())) # 独立行
                    if contact_node_set_name in odb_node_sets.keys(): # if 块开始
                         try: # try 块开始
                             found_odb_node_set = odb_node_sets[contact_node_set_name] # 独立行
                             log_to_excel("INFO", "MATCH FOUND: ODB NodeSet '{}' accessed by name.".format(contact_node_set_name)) # 独立行
                         except Exception as e_get_odb_ns: # except 块开始
                              log_to_excel("WARNING", "Found key '{}' but failed to get ODB NodeSet object.".format(contact_node_set_name), exc_obj=e_get_odb_ns) # 独立行
                    else: # else 块开始
                         log_to_excel("WARNING", "Could not find NodeSet '{}' by name in ODB keys.".format(contact_node_set_name)) # 独立行

                if found_odb_node_set is None: log_to_excel("ERROR", "Could not access NodeSet named '{}' in ODB.".format(contact_node_set_name)); log_to_excel("WARNING", "Skipping ODB extraction."); node_set_region = None # 独立行 (分号允许)
                else:
                    node_set_region = found_odb_node_set; log_to_excel("INFO", "Accessed NodeSet object: {}".format(node_set_region.name)) # 独立行 (分号允许)
                    try: log_to_excel("DEBUG", "Nodes in NodeSet: {}".format(len(node_set_region.nodes))) # 独立行
                    except Exception as count_e: log_to_excel("WARNING", "Could not get node count.", exc_obj=count_e) # 独立行
                # --- ODB NodeSet 获取结束 ---
                # Access last frame
                analysis_step = None; last_frame = None # 独立行 (分号允许)
                if node_set_region: #... get frame logic ...
                    log_to_excel("DEBUG", "Accessing step: {}".format(step_name)) # 独立行
                    try:
                        analysis_step = odb.steps[step_name]; log_to_excel("DEBUG", "Step '{}' accessed.".format(step_name)) # 独立行 (分号允许)
                        if analysis_step.frames: last_frame = analysis_step.frames[-1]; log_to_excel("DEBUG", "Accessed last frame (ID: {})".format(last_frame.frameId)) # 独立行 (分号允许)
                        else: log_to_excel("WARNING", "Step '{}' has no frames.".format(step_name)); last_frame = None # 独立行 (分号允许)
                    except KeyError as e: log_to_excel("ERROR", "Step '{}' not found.".format(step_name), exc_obj=e); last_frame = None # 独立行 (分号允许)
                    except IndexError as e: log_to_excel("ERROR", "Error accessing frames.", exc_obj=e); last_frame = None # 独立行 (分号允许)
                else: log_to_excel("WARNING", "Skipping step/frame access.") # 独立行
                # Extract Max Values
                if node_set_region and last_frame: #... extract loop ...
                    log_to_excel("DEBUG", "Extracting max values...") # 独立行
                    for variable_name in output_variables:
                         log_to_excel("DEBUG", "--- Var: {} ---".format(variable_name)); max_val = 0.0; data_found = False; subset=None # 独立行 (分号允许)
                         try: #... var processing ...
                             fieldOutput = last_frame.fieldOutputs[variable_name] # 独立行
                             subset = fieldOutput.getSubset(region=node_set_region, position=NODAL) # Use found object
                             if subset is None: log_to_excel("WARNING","Subset is None."); max_val=0.0; data_found=False # 独立行 (分号允许)
                             else: #... bulk/values ...
                                 if hasattr(subset, 'bulkDataBlocks') and subset.bulkDataBlocks:
                                     try:
                                         block_data = subset.bulkDataBlocks[0].data # 独立行
                                         if block_data is not None and len(block_data) > 0:
                                             data_array = np.array(block_data).flatten() # 独立行
                                             if data_array.size > 0: max_val = np.max(data_array); data_found = True; log_to_excel("INFO", "Max {} (bulk): {:.4e}".format(variable_name, max_val)) # 独立行 (分号允许)
                                             else: log_to_excel("WARNING", "bulkData empty '{}'.".format(variable_name)) # 独立行
                                         else: log_to_excel("WARNING", "bulkData[0].data empty '{}'.".format(variable_name)) # 独立行
                                     except Exception as bulk_err: log_to_excel("ERROR", "Error processing bulkData for {}.".format(variable_name), exc_obj=bulk_err) # 独立行
                                 if not data_found and hasattr(subset, 'values') and subset.values:
                                     try:
                                         valid_data = [fv.data for fv in subset.values if hasattr(fv, 'data') and isinstance(fv.data, (int, float))] # 独立行
                                         if valid_data: max_val = max(valid_data); data_found = True; log_to_excel("INFO", "Max {} (values): {:.4e}".format(variable_name, max_val)) # 独立行 (分号允许)
                                         else: log_to_excel("WARNING", "No valid numeric data in values for {}.".format(variable_name)) # 独立行
                                     except Exception as val_err: log_to_excel("ERROR", "Error iterating values for {}.".format(variable_name), exc_obj=val_err) # 独立行
                                 if not data_found: log_to_excel("WARNING", "Could not extract data for {}. Max=0.0.".format(variable_name)); max_val = 0.0 # 独立行 (分号允许)
                         except KeyError as e: log_to_excel("WARNING", "Field output '{}' not found.".format(variable_name)); max_val = 0.0 # 独立行 (分号允许)
                         except Exception as e: log_to_excel("ERROR", "Unexpected error processing {}. Max=0.0.".format(variable_name), exc_obj=e); max_val = 0.0 # 独立行 (分号允许)
                         log_to_excel("DEBUG", "Storing final max {:.4e} for '{}'".format(max_val, variable_name)) # 独立行
                         max_values_for_frame[variable_name] = max_val # 独立行
                    log_to_excel("DEBUG", "Finished extracting max values.") # 独立行
                else: log_to_excel("WARNING", "Skipping extraction loop.") # 独立行
            # Outer ODB except/finally (不变)
            except odbAccess.OdbError as e:
                log_to_excel("ERROR", "OdbError opening/processing ODB '{}'.".format(odb_path), exc_obj=e) # 独立行
                log_to_excel("WARNING", "Skipping results...") # 独立行
            except Exception as e:
                log_to_excel("ERROR", "Unexpected error during ODB processing for '{}'.".format(odb_path), exc_obj=e) # 独立行
                log_to_excel("WARNING", "Skipping results...") # 独立行
            finally:
                 if odb: # if 块开始
                     try: # try 块开始
                         odb.close() # 独立行
                         log_to_excel("DEBUG", "Closed ODB: {}".format(odb_path)) # 独立行
                     except Exception as close_e: # except 块开始
                         log_to_excel("ERROR", "Error closing ODB '{}'.".format(odb_path), exc_obj=close_e) # 独立行
                 else: # else 块开始
                     log_to_excel("DEBUG", "ODB not opened or error before closing.") # 独立行


            # --- Write results ---
            # ... (代码不变, 确保 except 分行) ...
            log_to_excel("DEBUG", "Preparing row data for Result Excel...")
            row_data = [pressure_iter_num, current_pressure, x_iter_num, current_x_offset, z_iter_num, current_z_offset];
            for var in output_variables: row_data.append(max_values_for_frame.get(var, 0.0))
            log_to_excel("DEBUG", "Attempting to write row to Result Excel: {}".format(row_data))
            try:
                if ws_result: ws_result.append(row_data); log_to_excel("DEBUG", "Row appended to Result Excel.")
                else: log_to_excel("ERROR","Result worksheet unavailable.")
            except Exception as e: log_to_excel("ERROR", "Failed to append data to Result Excel.", exc_obj=e)


            log_to_excel("INFO", "------- Z Iteration {} completed -------".format(z_iter_num)) # 独立行
            # --- Z 循环结束 ---
        log_to_excel("INFO", "----- X Iteration {} completed -----".format(x_iter_num)) # 独立行
        # --- X 循环结束 ---
    log_to_excel("INFO", "--- Pressure Iteration {} completed ---".format(pressure_iter_num)) # 独立行
# --- 压力循环结束 ---

# =============================================================================
# Script Finalization
# =============================================================================
# ... (代码不变, 确保 except 分行) ...
log_to_excel("INFO", "Script execution finished main loop.")
log_to_excel("INFO", "Attempting to save final Result Excel workbook to: {}".format(output_excel_file))
try:
    if wb_result:
        wb_result.save(filename=output_excel_file) # 独立行
        log_to_excel("INFO", "=====================================================") # 独立行
        log_to_excel("INFO", "Script finished successfully (Result Excel saved)!") # 独立行
        log_to_excel("INFO", "All iterations appear completed.") # 独立行
        log_to_excel("INFO", "Results potentially saved to: {}".format(os.path.abspath(output_excel_file))) # 独立行
        log_to_excel("INFO", "=====================================================") # 独立行
        log_to_excel("DEBUG", "Result Excel workbook saved successfully.") # 独立行
    else:
         log_to_excel("ERROR","Result workbook unavailable.") # 独立行
except Exception as e:
    log_to_excel("ERROR", "Failed to save Result Excel '{}'...".format(output_excel_file), exc_obj=e) # 独立行
    log_to_excel("DEBUG", "Error occurred during final Result Excel save.") # 独立行
log_to_excel("INFO", "Attempting to save Debug Log Excel file...")
saved_debug = save_debug_log()
log_to_excel("INFO", "Script execution complete. Debug log save status: {}".format(saved_debug))
if not saved_debug:
    log_to_excel("INFO", "Retrying to save Debug Log Excel file at the very end...") # 独立行
    save_debug_log() # 独立行
# 脚本结束