# -*- coding: utf-8 -*-

# =============================================================================
# Abaqus Python Script for Parametric Pressure Study
# Extracts MAX CPRESS, CSHEAR1, CSHEAR2 from a Contact Node Set
# Logs Debug/Error Messages to a Separate Excel File
# Compatible with older Python versions (pre-3.6, including print statement)
# Uses Node Set for region specification.
# Corrected structure for except blocks and early print statements.
# =============================================================================
#
# Description: (Same as before)
# Author: [Your Name/Organization] - Modified based on user request
# Date: [Date]
# Version: 3.4 (Python 2 Print Compatibility) # <<< 版本号更新
# Prerequisites: (Same as before)
#

# =============================================================================
# Import necessary modules
# =============================================================================
import os
import sys
import datetime
import traceback

# --- 初始化 Debug Excel 日志 ---
target_directory = r'C:\Users\admin\Desktop\FEM'
debug_excel_file_name = 'Debug_Log.xlsx'
debug_excel_file = os.path.join(target_directory, debug_excel_file_name)
initial_log_success = False

# 确保目录存在 (兼容旧Python)
if not os.path.exists(target_directory):
    try:
        os.makedirs(target_directory)
        # <<< 修改: 使用 Python 2 print 语句 >>>
        print "DEBUG: Created target directory: {}".format(target_directory)
    except OSError as e:
        # <<< 修改: 使用 Python 2 print 语句, 分行 >>>
        print "FATAL ERROR: Cannot create target directory for logging: {}. Error: {}".format(target_directory, e)
        sys.exit("Script aborted: cannot create log directory.")
    except Exception as e_create:
         # <<< 修改: 使用 Python 2 print 语句, 分行 >>>
         print "FATAL ERROR: Unexpected error creating directory: {}. Error: {}".format(target_directory, e_create)
         sys.exit("Script aborted: unexpected directory creation error.")

# 尝试导入 openpyxl 并初始化 Debug Excel
# <<< 修改: 调整 except 块结构, 使用 Python 2 print >>>
try:
    import openpyxl
    from openpyxl import Workbook
    wb_debug = Workbook()
    ws_debug = wb_debug.active
    ws_debug.title = 'DebugLog'
    ws_debug.append(['Timestamp', 'Level', 'Message', 'Traceback'])
    initial_log_success = True
except ImportError:
    # <<< 修改: 使用 Python 2 print 语句, 分行 >>>
    print "FATAL ERROR: The 'openpyxl' library is required for logging but not found."
    print "Please install it in your Abaqus Python environment."
    print "Run: abaqus python -m pip install openpyxl"
    sys.exit("Script aborted due to missing openpyxl for logging.")
except Exception as e:
    # <<< 修改: 使用 Python 2 print 语句, 分行 >>>
    print "FATAL ERROR: Failed to initialize Debug Excel Log ({}). Error: {}".format(debug_excel_file, e)
    sys.exit("Script aborted: Cannot initialize debug log.")

# --- 定义日志记录函数 ---
def log_to_excel(level, message, exc_obj=None):
    if not initial_log_success: return
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tb_str = ""
        if exc_obj:
            try:
                tb_str = "".join(traceback.format_exception(type(exc_obj), exc_obj, exc_obj.__traceback__))
            except Exception as format_exc_err:
                tb_str = "Traceback formatting failed (Error: {}). Original Exception: {}".format(format_exc_err, exc_obj)
        ws_debug.append([str(timestamp), str(level), str(message), str(tb_str)])
    except Exception as log_e:
        # <<< 修改: 使用 Python 2 print 语句 >>>
        print "ERROR writing to debug log: {}. Original message: [{}] {}".format(log_e, level, message)

# --- 定义保存 Debug Excel 的函数 ---
def save_debug_log():
    if not initial_log_success: return False
    try:
        wb_debug.save(debug_excel_file)
        # <<< 修改: 使用 Python 2 print 语句 >>>
        print "Attempting to save debug log to {}".format(debug_excel_file)
        return True
    except Exception as save_e:
        # <<< 修改: 使用 Python 2 print 语句 >>>
        print "ERROR: Failed to save debug log file '{}'. Error: {}".format(debug_excel_file, save_e)
        return False

# --- 开始记录日志 ---
log_to_excel("DEBUG", "Importing Abaqus modules...")
from abaqus import *
from abaqusConstants import *
import odbAccess
import job
import load
import step

# <<< 修改: 调整 except 块结构 >>>
try:
    import numpy as np
    log_to_excel("DEBUG", "Numpy imported successfully.")
except ImportError as e:
    err_msg = "The 'numpy' library is required..." # 省略
    log_to_excel("ERROR", err_msg, exc_obj=e)
    save_debug_log()
    sys.exit("Script aborted due to missing numpy.") # 确保在独立行

log_to_excel("DEBUG", "Standard Python modules imported.")

# =============================================================================
# USER CONFIGURABLE PARAMETERS - MODIFY THESE VALUES
# =============================================================================
log_to_excel("DEBUG", "Setting user configurable parameters...")
cae_file_path = r'D:\softgripper2.cae'
model_name = 'Model-1'
pressure_load_names = [
    'pressure1', 'pressure2', 'pressure3', 'pressure4a', 'pressure4b', 'pressure4c',
    'pressure4d', 'pressure4e', 'pressure4f', 'pressure5a', 'pressure5b', 'pressure5c',
    'pressure5d', 'pressure5e'
]
num_iterations = 2
initial_pressure = 10000.0
pressure_increment = 1000.0
excel_file_name = 'Max_Contact_Stress_Results.xlsx'
output_excel_file = os.path.join(target_directory, excel_file_name)

# --- 使用 Node Set ---
# !!! --- 请确保这个名称与你在 Abaqus/CAE 中定义的节点集名称完全一致（包括大小写） --- !!!
contact_node_set_name = 'NS_CONTACT_MAIN' # <<< 在这里填入你定义的节点集名称
# !!! -------------------------------------------------------------------------- !!!

step_name = 'Bending'
output_variables = ["CPRESS", "CSHEAR1", "CSHEAR2"]
log_to_excel("DEBUG", "User parameters set.")
log_to_excel("DEBUG", "Target output directory: {}".format(target_directory))
log_to_excel("DEBUG", "Output Result file path: {}".format(output_excel_file))
log_to_excel("DEBUG", "Debug Excel file path: {}".format(debug_excel_file))
log_to_excel("DEBUG", "Using Node Set for region: {}".format(contact_node_set_name))

# =============================================================================
# Script Initialization
# =============================================================================
log_to_excel("INFO", "Starting Abaqus parametric pressure study script...")

# --- Initialize RESULT Excel Workbook ---
log_to_excel("DEBUG", "Initializing Result Excel workbook...")
excel_headers = ["Iteration", "Pressure"] + ["Max " + var for var in output_variables]
wb_result = None; ws_result = None
# <<< 修改: 调整 except 块结构 >>>
try:
    wb_result = Workbook(); ws_result = wb_result.active; ws_result.title = 'Max Contact Data'
    log_to_excel("DEBUG", "Appending headers to Result Excel sheet...")
    ws_result.append(excel_headers)
    log_to_excel("DEBUG", "Result Excel headers written: {}".format(excel_headers))
except Exception as e:
    log_to_excel("ERROR", "Failed to initialize Result Excel workbook.", exc_obj=e)
    save_debug_log()
    sys.exit("Script aborted: Failed to initialize result excel.") # 独立行

# --- Open the CAE file and access the model ---
log_to_excel("DEBUG", "Attempting to open CAE file: {}".format(cae_file_path))
myModel = None
# <<< 修改: 调整 except 块结构 >>>
try:
    openMdb(pathName=cae_file_path); log_to_excel("INFO", "Successfully opened CAE file: {}".format(cae_file_path))
    myModel = mdb.models[model_name]; log_to_excel("INFO", "Accessed model: {}".format(model_name))
except MdbError as e:
    log_to_excel("ERROR", "Failed to open CAE file '{}'. MdbError.".format(cae_file_path), exc_obj=e)
    save_debug_log()
    sys.exit("Script aborted due to CAE file error.") # 独立行
except KeyError as e:
    log_to_excel("ERROR", "Model '{}' not found in CAE file '{}'. KeyError.".format(model_name, cae_file_path), exc_obj=e)
    save_debug_log()
    sys.exit("Script aborted due to model access error.") # 独立行
except Exception as e:
    log_to_excel("ERROR", "An unexpected error occurred opening CAE file or accessing model.", exc_obj=e)
    save_debug_log()
    sys.exit("Script aborted.") # 独立行

# --- 更改当前工作目录到目标文件夹 ---
log_to_excel("DEBUG", "Changing current working directory to: {}".format(target_directory))
# <<< 修改: 调整 except 块结构 >>>
try:
    os.chdir(target_directory)
    log_to_excel("INFO", "Current working directory changed successfully.")
except OSError as e:
    log_to_excel("ERROR", "Could not change working directory to '{}'. Please check path and permissions.".format(target_directory), exc_obj=e)
    save_debug_log()
    sys.exit("Script aborted due to chdir error.") # 独立行

# =============================================================================
# Main Simulation Loop
# =============================================================================
log_to_excel("INFO", "Entering main simulation loop...")
for i in range(num_iterations):
    iteration_number = i + 1
    log_to_excel("INFO", "------------------ Starting Iteration {} of {} ------------------".format(iteration_number, num_iterations))

    # --- 计算压力 ---
    log_to_excel("DEBUG", "Calculating current pressure...")
    current_pressure = initial_pressure + i * pressure_increment
    log_to_excel("INFO", "Current Pressure: {:.2f} Pa".format(current_pressure))

    # --- 修改载荷 ---
    log_to_excel("DEBUG", "Updating pressure loads in the model...")
    # <<< 修改: 调整 except 块结构 >>>
    try:
        if myModel is None: raise ValueError("MDB Model object (myModel) is not available.")
        for load_name in pressure_load_names:
            load_object = myModel.loads[load_name]
            load_object.setValues(magnitude=current_pressure)
        log_to_excel("INFO", "Successfully updated {} pressure loads.".format(len(pressure_load_names)))
    except KeyError as e:
        log_to_excel("ERROR", "Pressure load name '{}' not found...".format(e, model_name), exc_obj=e)
        save_debug_log()
        sys.exit("Script aborted due to load name error.") # 独立行
    except Exception as e:
        log_to_excel("ERROR", "Failed to update pressure loads.", exc_obj=e)
        save_debug_log()
        sys.exit("Script aborted.") # 独立行

    # --- 创建/运行 Job ---
    job_name = 'PressureStudy_Iter_{}'.format(iteration_number)
    log_to_excel("DEBUG", "Creating Abaqus job object: {}".format(job_name))
    myJob = None
    # <<< 修改: 调整 except 块结构 >>>
    try:
        if myModel is None: raise ValueError("MDB Model object (myModel) is not available for job creation.")
        myJob = mdb.Job(name=job_name, model=model_name, description='Pressure study iteration {}'.format(iteration_number), type=ANALYSIS, #...省略参数...
                        resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, numDomains=1, numGPUs=0)
        log_to_excel("DEBUG", "Job object created: {}".format(job_name))
        log_to_excel("INFO", "Submitting job: {}".format(job_name))
        myJob.submit(consistencyChecking=OFF)
        log_to_excel("DEBUG", "Job submission command issued for: {}".format(job_name))
        log_to_excel("INFO", "Waiting for job completion: {}...".format(job_name))
        myJob.waitForCompletion()
        log_to_excel("INFO", "Job {} completed.".format(job_name))
    except AbaqusException as e:
        log_to_excel("ERROR", "Abaqus job failed for {}. AbaqusException.".format(job_name), exc_obj=e)
        save_debug_log()
        sys.exit("Script aborted due to job failure.") # 独立行
    except Exception as e:
        log_to_excel("ERROR", "Unexpected error during job handling for {}.".format(job_name), exc_obj=e)
        save_debug_log()
        sys.exit("Script aborted.") # 独立行

    # --- Post-processing: Extract results from ODB ---
    odb_path = job_name + '.odb'
    log_to_excel("INFO", "Starting ODB post-processing for: {}".format(odb_path))
    odb = None
    max_values_for_frame = {var: 0.0 for var in output_variables}
    log_to_excel("DEBUG", "Initialized max_values_for_frame: {}".format(max_values_for_frame))

    # <<< 修改: 调整 except 块结构 >>>
    try:
        log_to_excel("DEBUG", "Attempting to open ODB file: {}".format(odb_path))
        if not os.path.exists(odb_path):
             err_msg = "ODB file not found at expected path: {}. Job might have failed silently.".format(odb_path)
             log_to_excel("ERROR", err_msg)
             raise odbAccess.OdbError(err_msg)
        odb = odbAccess.openOdb(path=odb_path, readOnly=True)
        log_to_excel("INFO", "Successfully opened ODB file: {}".format(odb_path))

        # --- 获取 Node Set region ---
        node_set_region = None
        log_to_excel("DEBUG", "Attempting to access NodeSet: '{}' at Assembly level.".format(contact_node_set_name))
        try:
            node_set_region = odb.rootAssembly.nodeSets[contact_node_set_name]
            log_to_excel("INFO", "Successfully accessed NodeSet region: {}".format(node_set_region.name))
            try:
                log_to_excel("DEBUG", "Number of nodes in NodeSet '{}': {}".format(node_set_region.name, len(node_set_region.nodes)))
            except:
                 log_to_excel("WARNING", "Could not determine number of nodes in the NodeSet.")
        except KeyError as e:
            log_to_excel("ERROR", "Could not find NodeSet '{}' at Assembly level. Please check name and definition in CAE/ODB.".format(contact_node_set_name), exc_obj=e)
            log_to_excel("WARNING", "Skipping ODB data extraction for this iteration due to NodeSet access error.")
            node_set_region = None
        except Exception as e_ns:
             log_to_excel("ERROR", "An error occurred while accessing NodeSet '{}'.".format(contact_node_set_name), exc_obj=e_ns)
             node_set_region = None

        # --- Access the last frame of the specified step ---
        analysis_step = None
        last_frame = None
        if node_set_region:
            log_to_excel("DEBUG", "Attempting to access step: {}".format(step_name))
            try:
                analysis_step = odb.steps[step_name]
                log_to_excel("DEBUG", "Step '{}' accessed.".format(step_name))
                if analysis_step.frames:
                    log_to_excel("DEBUG", "Accessing last frame of step '{}'...".format(step_name))
                    last_frame = analysis_step.frames[-1]
                    log_to_excel("DEBUG", "Accessed last frame (Frame ID: {}) of step: {}".format(last_frame.frameId, step_name))
                else:
                    log_to_excel("WARNING", "Step '{}' exists but contains no frames...".format(step_name, odb_path))
                    last_frame = None
            except KeyError as e:
                log_to_excel("ERROR", "Step '{}' not found...".format(step_name, odb_path), exc_obj=e)
                last_frame = None
            except IndexError as e:
                 log_to_excel("ERROR", "Error accessing frames for step '{}'...".format(step_name), exc_obj=e)
                 last_frame = None
        else:
            log_to_excel("WARNING", "Skipping step/frame access because NodeSet was not found/accessed.")


        # --- Extract Max Values for each variable ---
        if node_set_region and last_frame:
            log_to_excel("DEBUG", "Entering loop to extract max values for variables: {}".format(output_variables))
            for variable_name in output_variables:
                log_to_excel("DEBUG", "--- Processing variable: {} ---".format(variable_name))
                max_val = 0.0; fieldOutput = None; subset = None; position_to_try = NODAL
                log_to_excel("DEBUG", "Using position={} for getSubset.".format(position_to_try))
                # <<< 修改: 调整 except 块结构 >>>
                try:
                    log_to_excel("DEBUG", "Getting field output for '{}' from frame ID {}...".format(variable_name, last_frame.frameId))
                    fieldOutput = last_frame.fieldOutputs[variable_name]
                    log_to_excel("DEBUG", "Field output '{}' obtained. Type: {}".format(variable_name, type(fieldOutput)))

                    # 使用 NodeSet 获取子集
                    log_to_excel("DEBUG", "Getting subset for '{}' using NodeSet '{}'...".format(variable_name, node_set_region.name))
                    subset = fieldOutput.getSubset(region=node_set_region, position=position_to_try)
                    log_to_excel("DEBUG", "Subset obtained using NodeSet for '{}'. Type: {}".format(variable_name, type(subset)))

                    # 检查 subset 是否成功获取
                    if subset is None:
                        log_to_excel("WARNING", "Subset could not be obtained using NodeSet. Skipping data extraction for {}.".format(variable_name))
                        max_val = 0.0
                        data_found = False
                    else:
                        # 现有数据提取逻辑
                        data_found = False
                        log_to_excel("DEBUG", "Checking subset.bulkDataBlocks...")
                        if hasattr(subset, 'bulkDataBlocks') and subset.bulkDataBlocks:
                            log_to_excel("DEBUG", "Number of bulkDataBlocks: {}".format(len(subset.bulkDataBlocks)))
                            try:
                                 block_data = subset.bulkDataBlocks[0].data
                                 log_to_excel("DEBUG", "Accessed bulkDataBlocks[0].data. Type: {}, Length/Size: {}".format(type(block_data), len(block_data) if hasattr(block_data, '__len__') else 'N/A'))
                                 if block_data is not None and len(block_data) > 0:
                                     data_array = np.array(block_data).flatten()
                                     log_to_excel("DEBUG", "Converted bulk data to numpy array. Shape: {}, Size: {}".format(data_array.shape, data_array.size))
                                     if data_array.size > 0:
                                         log_to_excel("DEBUG", "Calculating max value from bulkDataBlocks data...")
                                         max_val = np.max(data_array); data_found = True
                                         log_to_excel("INFO", "Max value for {} (from bulkDataBlocks): {:.4e}".format(variable_name, max_val))
                                     else: log_to_excel("WARNING", "bulkDataBlocks numpy array is empty for '{}'.".format(variable_name))
                                 else: log_to_excel("WARNING", "bulkDataBlocks[0].data is None or empty for '{}'.".format(variable_name))
                            except Exception as bulk_err: log_to_excel("ERROR", "Error processing bulkDataBlocks for {}.".format(variable_name), exc_obj=bulk_err)
                        else: log_to_excel("DEBUG", "bulkDataBlocks not found or empty.")
                        if not data_found:
                            log_to_excel("DEBUG", "Checking subset.values for '{}'...".format(variable_name))
                            if hasattr(subset, 'values') and subset.values:
                                log_to_excel("DEBUG", "Number of values in subset.values: {}".format(len(subset.values)))
                                try:
                                    log_to_excel("DEBUG", "Iterating through subset.values to find max numeric data...")
                                    if len(subset.values) > 0:
                                         log_to_excel("DEBUG", "First item in subset.values: type={}, value={}".format(type(subset.values[0]), subset.values[0]))
                                         if hasattr(subset.values[0], 'data'): log_to_excel("DEBUG", "First item has .data attribute: type={}, value={}".format(type(subset.values[0].data), subset.values[0].data))
                                    valid_data = [fv.data for fv in subset.values if hasattr(fv, 'data') and isinstance(fv.data, (int, float))]
                                    log_to_excel("DEBUG", "Found {} valid numeric data points in subset.values.".format(len(valid_data)))
                                    if valid_data:
                                         max_val = max(valid_data); data_found = True
                                         log_to_excel("INFO", "Max value for {} (from subset.values): {:.4e}".format(variable_name, max_val))
                                    else: log_to_excel("WARNING", "No valid numeric data found in subset.values for {}.".format(variable_name))
                                except Exception as val_err: log_to_excel("ERROR", "Error iterating through subset.values for {}.".format(variable_name), exc_obj=val_err)
                            else: log_to_excel("WARNING", "subset.values is also empty or does not exist for '{}'.".format(variable_name))
                        if not data_found:
                             log_to_excel("WARNING", "Could not extract valid data for {} using NodeSet region. Max value remains 0.0.".format(variable_name))
                             max_val = 0.0

                except KeyError as e:
                    log_to_excel("WARNING", "Field output '{}' not found in frame {}. Setting max to 0.0.".format(variable_name, last_frame.frameId))
                    max_val = 0.0
                except odbAccess.OdbError as e:
                    log_to_excel("ERROR", "An OdbAccess error occurred during getSubset for variable {} using NodeSet.".format(variable_name), exc_obj=e)
                    max_val = 0.0
                except AbaqusException as e:
                     log_to_excel("WARNING", "Abaqus error during getSubset/data access for {} using NodeSet.".format(variable_name), exc_obj=e)
                     max_val = 0.0
                except Exception as e:
                    log_to_excel("ERROR", "An unexpected error occurred processing {} using NodeSet. Setting max to 0.0.".format(variable_name), exc_obj=e)
                    max_val = 0.0

                log_to_excel("DEBUG", "Storing final max value {:.4e} for variable '{}'".format(max_val, variable_name))
                max_values_for_frame[variable_name] = max_val
            log_to_excel("DEBUG", "Finished extracting max values for all variables in this iteration.")
        else:
            log_to_excel("WARNING", "Skipping max value extraction loop because NodeSet region or last_frame is invalid/not found.")

    except odbAccess.OdbError as e:
        log_to_excel("ERROR", "An OdbError occurred opening ODB '{}'.".format(odb_path), exc_obj=e)
        log_to_excel("WARNING", "Skipping results extraction for this iteration due to OdbError.")
    except Exception as e:
        log_to_excel("ERROR", "An unexpected error occurred during ODB processing section for '{}'.".format(odb_path), exc_obj=e)
        log_to_excel("WARNING", "Skipping results extraction for this iteration due to unexpected exception.")
    finally:
        # <<< 修改: 调整 finally 块结构 >>>
        if odb:
            log_to_excel("DEBUG", "Attempting to close ODB file: {}".format(odb_path))
            try:
                odb.close()
                log_to_excel("DEBUG", "Closed ODB file: {}".format(odb_path))
            except Exception as close_e:
                log_to_excel("ERROR", "Error closing ODB file '{}'.".format(odb_path), exc_obj=close_e)
        else:
             log_to_excel("DEBUG", "ODB was not opened or an error occurred before closing.")

    # --- Write results for the current iteration to RESULT Excel ---
    log_to_excel("DEBUG", "Preparing row data for Result Excel...")
    row_data = [iteration_number, current_pressure];
    for var in output_variables: row_data.append(max_values_for_frame.get(var, 0.0))
    log_to_excel("DEBUG", "Attempting to write row to Result Excel: {}".format(row_data))
    # <<< 修改: 调整 except 块结构 >>>
    try:
        if ws_result:
            ws_result.append(row_data)
            log_to_excel("DEBUG", "Row successfully appended to Result Excel worksheet.")
        else:
            log_to_excel("ERROR","Result worksheet (ws_result) is not available. Cannot append data.")
    except Exception as e:
        log_to_excel("ERROR", "Failed to append data to Result Excel worksheet.", exc_obj=e)

    log_to_excel("INFO", "------------------ Iteration {} completed ------------------".format(iteration_number))

# =============================================================================
# Script Finalization
# =============================================================================
log_to_excel("INFO", "Script execution finished main loop.")

# --- 保存 Result Excel ---
log_to_excel("INFO", "Attempting to save final Result Excel workbook to: {}".format(output_excel_file))
# <<< 修改: 调整 except 块结构 >>>
try:
    if wb_result:
        wb_result.save(filename=output_excel_file)
        log_to_excel("INFO", "=====================================================")
        log_to_excel("INFO", "Script finished successfully (Result Excel saved)!")
        log_to_excel("INFO", "All {} iterations appear completed.".format(num_iterations))
        log_to_excel("INFO", "Results potentially saved to: {}".format(os.path.abspath(output_excel_file)))
        log_to_excel("INFO", "=====================================================")
        log_to_excel("DEBUG", "Result Excel workbook saved successfully.")
    else:
         log_to_excel("ERROR","Result workbook (wb_result) is not available. Cannot save.")
except Exception as e:
    log_to_excel("ERROR", "Failed to save the Result Excel file '{}'. Check permissions/if file is open.".format(output_excel_file), exc_obj=e)
    log_to_excel("DEBUG", "Error occurred during final Result Excel save.")

# --- 保存 Debug Excel ---
log_to_excel("INFO", "Attempting to save Debug Log Excel file...")
saved_debug = save_debug_log()

# Optional: Close the MDB if desired
# ... (省略) ...

log_to_excel("INFO", "Script execution complete. Debug log save status: {}".format(saved_debug))

# --- 最后再次尝试保存 Debug Log ---
if not saved_debug:
    log_to_excel("INFO", "Retrying to save Debug Log Excel file at the very end...")
    save_debug_log()

# 脚本结束