# -*- coding: utf-8 -*-

# =============================================================================
# Abaqus Python Script for Parametric Pressure Study (Simplified for Displacement Only)
# Version: 6.4.1 (Based on V6.4, Simplified for Displacement) # <<< 版本号更新
# Extracts Nodal Displacements (from disp_node_set_name) using NumPy check.
# Uses Node Sets for region specification. Strict syntax formatting applied.
# =============================================================================
#
# Description: Performs pressure iterations and extracts nodal displacement results.
# Author: [Your Name/Organization] - Modified based on user request
# Date: [Date]
# Prerequisites: (Updated)
#

# =============================================================================
# Import necessary modules
# =============================================================================
import os
import sys
import datetime
import traceback
# import math # 不需要 math

# --- 初始化 Debug Excel 日志 ---
target_directory = r'D:\FEM' # <<< 确认日志输出目录
debug_excel_file_name = 'Debug_Log_Displacement_Only.xlsx' # <<< 修改调试日志文件名
debug_excel_file = os.path.join(target_directory, debug_excel_file_name)
initial_log_success = False
wb_debug = None
ws_debug = None
log_init_dir_msg = None # 初始化

# --- 确保目录存在 (V6.4 严格结构) ---
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

# --- 尝试导入 openpyxl 并初始化 Debug Excel (V6.4 严格结构) ---
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
        print "Attempting to save debug log to {}".format(debug_excel_file) # 独立行
        wb_debug.save(debug_excel_file) # 独立行
        return True # 独立行
    # except 块开始
    except Exception as save_e:
        print "ERROR: Failed to save debug log file '{}'. Error: {}".format(debug_excel_file, save_e) # 独立行
        return False # 独立行

# --- 开始记录日志 ---
if initial_log_success and log_init_dir_msg: # 独立行
    log_to_excel("INFO", log_init_dir_msg.replace("DEBUG: ", "Directory Info: ")) # 独立行
elif initial_log_success: # 独立行
    log_to_excel("INFO", "Debug Excel log active.") # 独立行

log_to_excel("DEBUG", "Importing Abaqus modules...")
from abaqus import *
from abaqusConstants import *
import odbAccess
import job
import load
import step
import assembly # 确保导入

# 尝试导入 numpy (严格分行)
try:
    import numpy as np
    log_to_excel("DEBUG", "Numpy imported successfully.")
except ImportError as e:
    err_msg = "The 'numpy' library is required but not found. Please install it (e.g., 'abaqus python -m pip install numpy')."
    log_to_excel("ERROR", err_msg, exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted due to missing numpy.") # 独立行

log_to_excel("DEBUG", "Standard Python modules imported.")

# =============================================================================
# USER CONFIGURABLE PARAMETERS - MODIFY THESE VALUES
# =============================================================================
log_to_excel("DEBUG", "Setting user configurable parameters...")
cae_file_path = r'D:\softgripper1.cae' # <<< 确认 CAE 文件路径正确
model_name = 'Model-1' # <<< 确认模型名称
pressure_load_names = [ # <<< 确认压力载荷名称列表正确
    'pressure1', 'pressure2', 'pressure3', 'pressure4a', 'pressure4b', 'pressure4c',
    'pressure4d', 'pressure4e', 'pressure4f', 'pressure5a', 'pressure5b', 'pressure5c',
    'pressure5d', 'pressure5e'
]
num_iterations = 1 # <<< 确认压力迭代次数
initial_pressure = 25000.0 # <<< 确认初始压力 (Pa)
pressure_increment = 1000.0 # <<< 确认压力增量 (Pa)
excel_file_name = 'Displacement_Results.xlsx' # <<< 修改了结果文件名
output_excel_file = os.path.join(target_directory, excel_file_name) # 结果文件完整路径

# --- ODB Result Extraction Parameters ---
# !!! --- 请确保名称与你在 Abaqus/CAE 中定义的节点集名称完全一致 --- !!!
disp_node_set_name = 'EDGE_FOR_DISP'   # <<< 用于提取位移的节点集 (软体)
# !!! ----------------------------------- !!!

step_name = 'Bending' # <<< 确认分析步名称
displacement_variable_name = 'U' # 位移变量名 (通常是 'U')

log_to_excel("DEBUG", "User parameters set.")
log_to_excel("DEBUG", "Target output directory: {}".format(target_directory))
log_to_excel("DEBUG", "Output Excel file path: {}".format(output_excel_file))
log_to_excel("DEBUG", "Debug Excel file path: {}".format(debug_excel_file))
log_to_excel("DEBUG", "Pressure Iterations: {}".format(num_iterations))
log_to_excel("DEBUG", "Node Set for Displacement: {}".format(disp_node_set_name))
log_to_excel("DEBUG", "Displacement variable: {}".format(displacement_variable_name))


# =============================================================================
# Script Initialization
# =============================================================================
log_to_excel("INFO", "Starting Abaqus parametric pressure study script (Displacement Only)...")

# --- Initialize RESULT Excel Workbook with ONE Sheet ---
log_to_excel("DEBUG", "Initializing Result Excel workbook...")
disp_excel_headers = ['Iteration', 'Pressure', 'NodeLabel', 'X0', 'Y0', 'Z0', 'U1', 'U2', 'U3'] # <<< 位移表头
wb_result = None
ws_disp = None   # <<< 位移 Sheet
try:
    wb_result = Workbook() # 独立行
    # 创建位移 Sheet
    ws_disp = wb_result.active # 独立行 (默认第一个 sheet 就是 active)
    ws_disp.title = 'Nodal Displacements' # 独立行
    log_to_excel("DEBUG", "Appending headers to Displacement sheet...") # 独立行
    ws_disp.append(disp_excel_headers) # <<< 写入位移表头
    log_to_excel("DEBUG", "Displacement Excel headers written: {}".format(disp_excel_headers)) # 独立行
except Exception as e:
    log_to_excel("ERROR", "Failed to initialize Result Excel workbook or sheet.", exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted: Failed to initialize result excel.") # 独立行


# --- Open the CAE file and access the model ---
log_to_excel("DEBUG", "Attempting to open CAE file: {}".format(cae_file_path))
myModel = None
myAssembly = None # <<< 新增: 获取 Assembly 对象
initial_coords_dict = {} # <<< 新增: 存储初始坐标
try:
    openMdb(pathName=cae_file_path) # 独立行
    log_to_excel("INFO", "Successfully opened CAE file: {}".format(cae_file_path)) # 独立行
    myModel = mdb.models[model_name] # 独立行
    log_to_excel("INFO", "Accessed model: {}".format(model_name)) # 独立行
    myAssembly = myModel.rootAssembly # <<< 获取 Assembly 对象
    log_to_excel("INFO", "Accessed root assembly.") # 独立行

    # --- <<< 获取位移节点集内节点的初始坐标 >>> ---
    log_to_excel("DEBUG", "Attempting to get initial coordinates for nodes in Node Set '{}' from MDB...".format(disp_node_set_name)) # <<< 使用 disp_node_set_name
    if myAssembly: # 检查 Assembly 对象是否存在
        try: # try 块开始
            # 假设节点集在 Assembly 层面定义
            node_set_mdb = myAssembly.sets[disp_node_set_name] # <<< 使用 disp_node_set_name
            log_to_excel("DEBUG", "Accessed Node Set '{}' from MDB.".format(disp_node_set_name)) # 独立行
            # 获取节点对象列表
            nodes_in_set_mdb = node_set_mdb.nodes # 独立行
            log_to_excel("DEBUG", "Found {} nodes in Node Set '{}' (MDB).".format(len(nodes_in_set_mdb), disp_node_set_name)) # 独立行
            # 遍历节点并存储坐标
            node_count = 0 # 计数器
            for node in nodes_in_set_mdb: # for 循环开始
                try: # try 块开始 (获取单个节点坐标)
                    initial_coords_dict[node.label] = node.coordinates # 独立行
                    node_count += 1 # 独立行
                except Exception as e_coord: # except 块开始 (获取单个节点坐标失败)
                    log_to_excel("WARNING", "Could not get coordinates for node with label {} in MDB.".format(node.label), exc_obj=e_coord) # 独立行
            log_to_excel("INFO", "Stored initial coordinates for {} nodes from set '{}'.".format(node_count, disp_node_set_name)) # 独立行
        except KeyError as e_set: # except 块开始 (MDB中找不到节点集)
            log_to_excel("ERROR", "Node Set '{}' for displacement not found in the Assembly of the CAE file '{}'. Please check the name.".format(disp_node_set_name, cae_file_path), exc_obj=e_set) # 独立行
            save_debug_log() # 独立行
            sys.exit("Script aborted: Node Set for initial coordinates not found in MDB.") # 独立行
        except Exception as e_get_init: # except 块开始 (其他获取初始坐标错误)
             log_to_excel("ERROR", "An error occurred while getting initial node coordinates from MDB.", exc_obj=e_get_init) # 独立行
             save_debug_log() # 独立行
             sys.exit("Script aborted: Failed to get initial coordinates.") # 独立行
    else: # else 块开始 (Assembly 对象获取失败)
         log_to_excel("ERROR", "Could not access Assembly object from model '{}'.".format(model_name)) # 独立行
         save_debug_log() # 独立行
         sys.exit("Script aborted: Assembly object not found.") # 独立行
    # --- <<< 获取初始坐标结束 >>> ---

except MdbError as e: # except 块开始
    log_to_excel("ERROR", "Failed to open CAE file '{}'. MdbError.".format(cae_file_path), exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted.") # 独立行
except KeyError as e: # except 块开始 (处理 Model 或 NodeSet 名称错误)
    error_message = str(e) # 独立行
    key_error_source = "Unknown" # 独立行
    if model_name in error_message: # 独立行
        key_error_source = "Model '{}'".format(model_name) # 独立行
    elif disp_node_set_name in error_message and 'sets' in error_message: # 只检查位移节点集
        key_error_source = "Node Set '{}'".format(disp_node_set_name) # 独立行
    log_to_excel("ERROR", "KeyError accessing MDB object: {}. Details: {}".format(key_error_source, e), exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted.") # 独立行
except Exception as e: # except 块开始
    log_to_excel("ERROR", "An unexpected error occurred opening CAE file or accessing initial objects.", exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted.") # 独立行


# --- 更改当前工作目录到目标文件夹 ---
log_to_excel("DEBUG", "Changing current working directory to: {}".format(target_directory))
try:
    os.chdir(target_directory) # 独立行
    log_to_excel("INFO", "Current working directory changed successfully.") # 独立行
except OSError as e:
    log_to_excel("ERROR", "Could not change working directory to '{}'. Error: {}".format(target_directory, e), exc_obj=e) # 独立行
    save_debug_log() # 独立行
    sys.exit("Script aborted due to chdir error.") # 独立行

# =============================================================================
# Main Simulation Loop
# =============================================================================
log_to_excel("INFO", "Entering main simulation loop...")

for i in range(num_iterations):
    iteration_number = i + 1
    log_to_excel("INFO", "------------------ Starting Iteration {} of {} ------------------".format(iteration_number, num_iterations))

    # --- 计算压力 ---
    log_to_excel("DEBUG", "Calculating current pressure...") # 独立行
    current_pressure = initial_pressure + i * pressure_increment # 独立行
    log_to_excel("INFO", "Current Pressure: {:.2f} Pa".format(current_pressure)) # 独立行

    # --- 修改载荷 ---
    log_to_excel("DEBUG", "Updating pressure loads...") # 独立行
    try:
        if myModel is None:
            raise ValueError("MDB Model unavailable.") # 独立行
        for load_name in pressure_load_names:
            try: # 尝试获取并修改单个载荷
                load_object = myModel.loads[load_name] # 独立行
                load_object.setValues(magnitude=current_pressure) # 独立行
            except KeyError:
                 log_to_excel("WARNING", "Pressure load name '{}' not found in model. Skipping.".format(load_name)) # 记录警告，但继续
            except Exception as e_load_set:
                log_to_excel("ERROR", "Failed to set value for pressure load '{}'.".format(load_name), exc_obj=e_load_set) # 记录错误，但可能继续
                # 考虑是否需要在这里停止脚本
        log_to_excel("DEBUG", "Pressure loads update process completed (check warnings/errors).") # 独立行
    except ValueError as e: # Model 不可用
        log_to_excel("ERROR", str(e), exc_obj=e) # 独立行
        save_debug_log() # 独立行
        sys.exit("Script aborted.") # 独立行
    except Exception as e: # 其他意外错误
        log_to_excel("ERROR", "An unexpected error occurred while updating pressure loads.", exc_obj=e) # 独立行
        save_debug_log() # 独立行
        sys.exit("Script aborted.") # 独立行

    # --- 创建/运行 Job ---
    job_name = 'PressureStudy_Iter_{}'.format(iteration_number)
    log_to_excel("DEBUG", "Creating Abaqus job object: {}".format(job_name)) # 独立行
    myJob = None # 独立行
    try:
        if myModel is None:
            raise ValueError("MDB Model unavailable for Job creation.") # 独立行
        myJob = mdb.Job(name=job_name, model=model_name,
                        description='Pressure study iteration {}'.format(iteration_number),
                        type=ANALYSIS, # 假设是标准分析
                        resultsFormat=ODB, # 需要 ODB 用于后处理
                        multiprocessingMode=DEFAULT, # 使用默认并行设置
                        numCpus=1, numDomains=1, # 根据需要调整 CPU 数量
                        numGPUs=0) # 假设不使用 GPU
        log_to_excel("DEBUG", "Job object created.") # 独立行
        log_to_excel("INFO", "Submitting job: {}".format(job_name)) # 独立行
        myJob.submit(consistencyChecking=OFF) # 独立行
        log_to_excel("DEBUG", "Job submission issued.") # 独立行
        log_to_excel("INFO", "Waiting for job completion: {}...".format(job_name)) # 独立行
        myJob.waitForCompletion() # 独立行
        log_to_excel("INFO", "Job {} completed.".format(job_name)) # 独立行
    except AbaqusException as e:
        log_to_excel("ERROR", "Abaqus job failed for {}. Check '.msg' or '.dat' file for details.".format(job_name), exc_obj=e) # 独立行
        log_to_excel("WARNING", "Skipping results extraction for this iteration due to job failure.") # 独立行
        continue # 继续下一次迭代
    except ValueError as e: # Model 不可用
        log_to_excel("ERROR", str(e), exc_obj=e) # 独立行
        save_debug_log() # 独立行
        sys.exit("Script aborted during job creation.") # 独立行
    except Exception as e:
        log_to_excel("ERROR", "Unexpected error during job creation or execution for {}.".format(job_name), exc_obj=e) # 独立行
        log_to_excel("WARNING", "Skipping results extraction for this iteration due to unexpected job error.") # 独立行
        continue # 继续下一次迭代

    # --- Post-processing: Extract results from ODB ---
    odb_path = job_name + '.odb' # 独立行
    log_to_excel("INFO", "Starting ODB post-processing for: {}".format(odb_path)) # 独立行
    odb = None # 独立行

    try: # ODB Main Try
        log_to_excel("DEBUG", "Attempting to open ODB file: {}".format(odb_path)) # 独立行
        odb_exists = os.path.exists(odb_path) # 独立行
        if not odb_exists: # if 块开始
             err_msg = "ODB file not found: {}. Job might have failed or was deleted.".format(odb_path)
             log_to_excel("ERROR", err_msg) # 独立行
             log_to_excel("WARNING", "Skipping ODB processing for this iteration.") # 独立行
             continue # 跳过此迭代的 ODB 处理
        # else 块开始 (ODB 文件存在)
        odb = odbAccess.openOdb(path=odb_path, readOnly=True) # 独立行
        log_to_excel("INFO", "Successfully opened ODB file: {}".format(odb_path)) # 独立行

        # --- 获取位移节点集对象 ---
        node_set_region_disp = None   # 位移区域对象
        try:
            odb_node_sets = odb.rootAssembly.nodeSets # 获取一次即可
            log_to_excel("DEBUG", "Attempting to access Displacement NodeSet: '{}' in ODB...".format(disp_node_set_name)) # 独立行
            if disp_node_set_name in odb_node_sets.keys(): # if 块开始
                node_set_region_disp = odb_node_sets[disp_node_set_name] # 独立行
                log_to_excel("INFO", "Accessed Displacement NodeSet: {}".format(disp_node_set_name)) # 独立行
            else: # else 块开始
                log_to_excel("ERROR", "Could not find Displacement NodeSet '{}' in ODB '{}'. Cannot extract displacements.".format(disp_node_set_name, odb_path)) # 独立行
                node_set_region_disp = None # 明确设为 None
        except Exception as e_disp_ns: # Disp NodeSet Except
            log_to_excel("ERROR", "Error accessing Displacement NodeSet '{}' in ODB '{}'.".format(disp_node_set_name, odb_path), exc_obj=e_disp_ns) # 独立行
            node_set_region_disp = None # 明确设为 None


        # --- Access the last frame ---
        analysis_step = None # 独立行
        last_frame = None # 独立行
        if node_set_region_disp: # 仅当位移节点集有效时才需要访问帧
            log_to_excel("DEBUG", "Attempting to access step: {}".format(step_name)) # 独立行
            try: # Step/Frame Try
                analysis_step = odb.steps[step_name] # 独立行
                log_to_excel("DEBUG", "Step '{}' accessed.".format(step_name)) # 独立行
                if analysis_step.frames: # if 块开始
                    last_frame = analysis_step.frames[-1] # 独立行
                    log_to_excel("DEBUG", "Accessed last frame (ID: {})".format(last_frame.frameId)) # 独立行
                else: # else 块开始
                    log_to_excel("WARNING", "Step '{}' has no frames in ODB '{}'.".format(step_name, odb_path)) # 独立行
                    last_frame = None # 独立行
            except KeyError as e: # except 块开始
                log_to_excel("ERROR", "Step '{}' not found in ODB '{}'.".format(step_name, odb_path), exc_obj=e) # 独立行
                last_frame = None # 独立行
            except IndexError as e: # except 块开始
                log_to_excel("ERROR", "Error accessing frames in step '{}' of ODB '{}'.".format(step_name, odb_path), exc_obj=e) # 独立行
                last_frame = None # 独立行
            except Exception as e_step_frame:
                log_to_excel("ERROR", "Unexpected error accessing step/frame in ODB '{}'.".format(odb_path), exc_obj=e_step_frame)
                last_frame = None
        else: # else 块开始
            log_to_excel("WARNING", "Skipping step/frame access: Displacement NodeSet is invalid.") # 独立行

        # --- 提取节点位移值 ---
        if last_frame and node_set_region_disp: # 确保帧和节点集都有效
            log_to_excel("DEBUG", "Extracting nodal displacement values ('{}') from NodeSet '{}'...".format(displacement_variable_name, disp_node_set_name)) # 独立行
            try: # try for displacement extraction
                disp_field = last_frame.fieldOutputs[displacement_variable_name] # 获取 'U' 场输出
                log_to_excel("DEBUG", "Field output '{}' obtained.".format(displacement_variable_name)) # 独立行
                # 获取 'U' 在节点集上的子集
                disp_subset = disp_field.getSubset(region=node_set_region_disp, position=NODAL) # <<< 使用 disp_node_set_region
                log_to_excel("DEBUG", "Subset obtained for displacement.") # 独立行

                if disp_subset and hasattr(disp_subset, 'values') and disp_subset.values: # 检查子集和 values 是否有效
                    log_to_excel("DEBUG", "Iterating through {} displacement values...".format(len(disp_subset.values))) # 独立行
                    # 遍历每个节点的值
                    nodes_processed_count = 0
                    for fv in disp_subset.values: # for 循环开始
                        try: # try for processing one FieldValue
                            node_label = fv.nodeLabel # 获取节点标签
                            # <<< V6.4 修改: 使用 NumPy 检查位移数据 >>>
                            if hasattr(fv, 'data'): # 首先检查 data 属性是否存在
                                try: # 尝试转换为 NumPy 数组并检查大小
                                    disp_array = np.array(fv.data) # 尝试转换
                                    if disp_array.size == 3: # 检查元素数量是否为 3
                                        u1 = disp_array.item(0) # 使用 .item() 获取标量值
                                        u2 = disp_array.item(1)
                                        u3 = disp_array.item(2)
                                        # 从 MDB 获取初始坐标
                                        initial_coords = initial_coords_dict.get(node_label, (None, None, None)) # 独立行
                                        x0 = initial_coords[0] # 独立行
                                        y0 = initial_coords[1] # 独立行
                                        z0 = initial_coords[2] # 独立行
                                        # 准备写入 Excel 的行数据
                                        disp_row = [iteration_number, current_pressure, node_label, x0, y0, z0, u1, u2, u3] # 独立行
                                        # 写入位移 Sheet
                                        if ws_disp: # 检查 ws_disp 是否有效
                                            ws_disp.append(disp_row) # 独立行
                                            nodes_processed_count += 1
                                        else: # 独立行
                                            log_to_excel("ERROR", "Displacement sheet (ws_disp) is not available! Cannot write data for node {}.".format(node_label)) # 独立行
                                    else: # else 块开始 (Numpy 数组大小不对)
                                        log_to_excel("WARNING", "Displacement data for node {} has unexpected size after numpy conversion: {}. Data: {}".format(node_label, disp_array.size, fv.data)) # 独立行
                                except Exception as np_conv_err: # except 块开始 (转换或处理 NumPy 数组失败)
                                     log_to_excel("WARNING", "Could not process displacement data for node {} as numpy array.".format(node_label), exc_obj=np_conv_err) # 独立行
                            else: # else 块开始 (fv 没有 data 属性)
                                log_to_excel("WARNING", "FieldValue object for node {} missing 'data' attribute.".format(node_label)) # 独立行
                        except AttributeError as e_fv_attr: # except 块开始 (访问 fv 属性失败)
                            log_to_excel("WARNING", "Error accessing attributes (nodeLabel/data) for a FieldValue object.", exc_obj=e_fv_attr) # 独立行
                        except Exception as e_fv_proc: # 处理单个 FieldValue 的其他错误
                            log_to_excel("WARNING", "Unexpected error processing displacement data for a node.", exc_obj=e_fv_proc)
                    log_to_excel("INFO", "Finished iterating displacement values. Successfully processed and wrote data for {} nodes.".format(nodes_processed_count)) # 独立行
                else: # else 块开始 (disp_subset 无效或无 values)
                    log_to_excel("WARNING", "Could not get valid displacement subset or values for NodeSet '{}' in ODB '{}'.".format(disp_node_set_name, odb_path)) # 独立行

            except KeyError: # except 块开始 (找不到 'U' 场输出)
                log_to_excel("ERROR", "Field output '{}' not found in ODB frame (Step: {}, Frame ID: {}). Cannot extract displacements.".format(displacement_variable_name, step_name, last_frame.frameId if last_frame else 'N/A')) # 独立行
            except Exception as e_disp: # except 块开始 (提取位移的其他错误)
                log_to_excel("ERROR", "An unexpected error occurred during displacement extraction from ODB '{}'.".format(odb_path), exc_obj=e_disp) # 独立行
            # --- 位移提取结束 ---
        elif not node_set_region_disp:
            log_to_excel("WARNING", "Skipping displacement extraction because displacement NodeSet was not found/accessed in ODB '{}'.".format(odb_path)) # 独立行
        elif not last_frame:
             log_to_excel("WARNING", "Skipping displacement extraction because the last frame object is invalid in ODB '{}'.".format(odb_path)) # 独立行


    # except blocks for outer ODB processing try (ODB 文件存在时的处理)
    except odbAccess.OdbError as e:
        log_to_excel("ERROR", "An OdbError occurred opening/processing ODB '{}'.".format(odb_path), exc_obj=e) # 独立行
        log_to_excel("WARNING", "Skipping remaining ODB processing for this iteration...") # 独立行
    except Exception as e:
        log_to_excel("ERROR", "Unexpected error during ODB processing for '{}'.".format(odb_path), exc_obj=e) # 独立行
        log_to_excel("WARNING", "Skipping remaining ODB processing for this iteration...") # 独立行
    # finally block for outer ODB processing try
    finally:
        if odb: # if 块开始
             log_to_excel("DEBUG", "Attempting to close ODB file: {}".format(odb_path)) # 独立行
             try: # 关闭 ODB 的 try
                 odb.close() # 独立行
                 log_to_excel("DEBUG", "Closed ODB file: {}".format(odb_path)) # 独立行
             except Exception as close_e: # 关闭 ODB 的 except
                 log_to_excel("ERROR", "Error closing ODB file '{}'.".format(odb_path), exc_obj=close_e) # 独立行
        else: # else 块开始
              log_to_excel("DEBUG", "ODB was not opened or an error occurred before closing attempt for '{}'.".format(odb_path)) # 独立行

    log_to_excel("INFO", "------------------ Iteration {} completed ------------------".format(iteration_number)) # 独立行

# =============================================================================
# Script Finalization
# =============================================================================
log_to_excel("INFO", "Script execution finished main loop.") # 独立行
# --- 保存 Result Excel (只包含位移 Sheet) ---
log_to_excel("INFO", "Attempting to save final Result Excel workbook to: {}".format(output_excel_file))
try:
    if wb_result and ws_disp: # 确保 Workbook 和 Sheet 都存在
        wb_result.save(filename=output_excel_file) # 独立行
        log_to_excel("INFO", "=====================================================") # 独立行
        log_to_excel("INFO", "Script finished successfully (Result Excel saved)!") # 独立行
        log_to_excel("INFO", "All {} iterations appear completed.".format(num_iterations)) # 独立行
        log_to_excel("INFO", "Displacement results potentially saved to: {}".format(os.path.abspath(output_excel_file))) # 独立行
        log_to_excel("INFO", "=====================================================") # 独立行
        log_to_excel("DEBUG", "Result Excel workbook saved successfully.") # 独立行
    elif not wb_result:
         log_to_excel("ERROR","Result workbook object (wb_result) is unavailable. Cannot save.") # 独立行
    else: # wb_result 存在但 ws_disp 不存在 (理论上不应发生)
        log_to_excel("ERROR","Displacement worksheet object (ws_disp) is unavailable within the workbook. Cannot save.") # 独立行
except Exception as e:
    log_to_excel("ERROR", "Failed to save Result Excel file '{}'. Check permissions and file path.".format(output_excel_file), exc_obj=e) # 独立行

# --- 保存 Debug Excel ---
log_to_excel("INFO", "Attempting to save Debug Log Excel file...") # 独立行
saved_debug = save_debug_log() # 独立行
log_to_excel("INFO", "Script execution complete. Debug log save status: {}".format(saved_debug)) # 独立行
# --- 最后再次尝试保存 Debug Log ---
if not saved_debug: # 独立行
    log_to_excel("INFO", "Retrying to save Debug Log Excel file at the very end...") # 独立行
    save_debug_log() # 独立行
# 脚本结束
print "INFO: Abaqus Python script execution finished." # 结束时在控制台打印信息