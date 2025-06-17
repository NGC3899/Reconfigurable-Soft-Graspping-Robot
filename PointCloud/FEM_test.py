# -*- coding: utf-8 -*-

# =============================================================================
# Abaqus Python Script for Parametric Pressure Study
# Extracts MAX CPRESS, CSHEAR1, CSHEAR2 from a Contact Slave Surface
# Includes Debug Print Statements and Output Directory Management
# =============================================================================
#
# Description:
# This script automates Abaqus simulations by incrementally increasing pressure.
# For each run, it extracts the maximum values of specified contact stress
# variables (CPRESS, CSHEAR1, CSHEAR2) from a defined slave surface in the ODB
# and logs results (iteration, pressure, max values) into an Excel file.
# All output files (Job files, Excel) are placed in a specified directory.
#
# Author: [Your Name/Organization] - Modified based on user request
# Date: [Date]
# Version: 2.3 (Final with Output Dir) # <<< 版本号更新
#
# Prerequisites:
# 1. A working Abaqus.cae model file.
# 2. The 'openpyxl' and 'numpy' Python libraries must be installed in the
#    Abaqus Python environment.
#    (Run: abaqus python -m pip install openpyxl numpy)
# 3. Field outputs CPRESS, CSHEAR1, CSHEAR2 must be requested for the
#    specified step ('Bending') and associated with the contact interaction.
#

# =============================================================================
# Import necessary modules
# =============================================================================
print("DEBUG: Importing modules...")
from abaqus import *
from abaqusConstants import *
import odbAccess # For ODB operations
import job
import load
import step
# Import other Abaqus modules if needed by your specific CAE operations

try:
    import numpy as np
    print("DEBUG: Numpy imported successfully.")
except ImportError:
    print("ERROR: The 'numpy' library is required but not found.")
    print("Please install it in your Abaqus Python environment.")
    print("Run: abaqus python -m pip install numpy")
    exit()

try:
    import openpyxl
    from openpyxl import Workbook
    print("DEBUG: Openpyxl imported successfully.")
except ImportError:
    print("ERROR: The 'openpyxl' library is required but not found.")
    print("Please install it in your Abaqus Python environment.")
    print("Run: abaqus python -m pip install openpyxl")
    exit()

import os # For path manipulation and directory operations <<< 确认导入 os
import sys # For exiting gracefully
print("DEBUG: Standard Python modules imported.")

# =============================================================================
# USER CONFIGURABLE PARAMETERS - MODIFY THESE VALUES
# =============================================================================
print("DEBUG: Setting user configurable parameters...")
# --- File and Model Identification ---
cae_file_path = r'D:\softgripper2.cae' # <<< 使用原始字符串 r''
model_name = 'Model-1'

# --- <<< 新增: 定义目标输出文件夹 >>> ---
target_directory = r'C:\Users\admin\Desktop\FEM'

# --- Load Control ---
pressure_load_names = [
    'pressure1', 'pressure2', 'pressure3',
    'pressure4a', 'pressure4b', 'pressure4c',
    'pressure4d', 'pressure4e', 'pressure4f',
    'pressure5a', 'pressure5b', 'pressure5c',
    'pressure5d', 'pressure5e'
]
num_iterations = 2
initial_pressure = 10000.0
pressure_increment = 1000.0

# --- Output Configuration ---
# <<< 修改: 构建 Excel 文件的完整绝对路径 >>>
excel_file_name = 'Max_Contact_Stress_Results.xlsx'
output_excel_file = os.path.join(target_directory, excel_file_name)

# --- ODB Result Extraction Parameters ---
instance_name = 'SoftBendingGripper-1'
contact_slave_surface_name = 's_Surf-15'
step_name = 'Bending'                 # <<< 已修正为 Bending
output_variables = ["CPRESS", "CSHEAR1", "CSHEAR2"]
print("DEBUG: User parameters set.")
print("DEBUG: Target output directory: {}".format(target_directory))
print("DEBUG: Output Excel file path: {}".format(output_excel_file))

# =============================================================================
# Script Initialization
# =============================================================================
print("Starting Abaqus parametric pressure study script...")

# --- <<< 新增: 确保目标文件夹存在 >>> ---
print("DEBUG: Checking target directory: {}".format(target_directory))
if not os.path.exists(target_directory):
    print("DEBUG: Directory does not exist. Attempting to create...")
    try:
        os.makedirs(target_directory) # 不使用 exist_ok 参数
        print("DEBUG: Successfully created directory: {}".format(target_directory))
    except OSError as e:
        # 捕获创建过程中可能发生的错误 (例如权限问题)
        print("ERROR: Could not create directory '{}'. Please check path and permissions.".format(target_directory))
        print(e)
        sys.exit("Script aborted due to directory creation error.")
else:
    print("DEBUG: Target directory already exists.")

# --- Initialize Excel Workbook ---
print("DEBUG: Initializing Excel workbook...")
excel_headers = ["Iteration", "Pressure"] + ["Max " + var for var in output_variables]
wb = Workbook()
ws = wb.active
ws.title = 'Max Contact Data'
print("DEBUG: Appending headers to Excel sheet...")
ws.append(excel_headers)
print("DEBUG: Excel headers written: {}".format(excel_headers))

# --- Open the CAE file and access the model ---
# 在改变工作目录前打开 MDB
print("DEBUG: Attempting to open CAE file: {}".format(cae_file_path))
try:
    openMdb(pathName=cae_file_path)
    print("DEBUG: Successfully opened CAE file: {}".format(cae_file_path))
    myModel = mdb.models[model_name]
    print("DEBUG: Accessed model: {}".format(model_name))
except MdbError as e:
    print("ERROR: Failed to open CAE file '{}'.".format(cae_file_path))
    print(e)
    sys.exit("Script aborted due to CAE file error.")
except KeyError as e:
    print("ERROR: Model '{}' not found in CAE file '{}'.".format(model_name, cae_file_path))
    print(e)
    sys.exit("Script aborted due to model access error.")
except Exception as e:
    print("ERROR: An unexpected error occurred opening CAE file or accessing model.")
    print(e)
    sys.exit("Script aborted.")

# --- <<< 新增: 更改当前工作目录到目标文件夹 >>> ---
# Job 文件将在此目录下创建
print("DEBUG: Changing current working directory to: {}".format(target_directory))
try:
    os.chdir(target_directory)
    print("DEBUG: Current working directory changed successfully.")
except OSError as e:
    print("ERROR: Could not change working directory to '{}'. Please check path and permissions.".format(target_directory))
    print(e)
    sys.exit("Script aborted due to chdir error.")

# =============================================================================
# Main Simulation Loop
# =============================================================================
print("DEBUG: Entering main simulation loop...")
for i in range(num_iterations):
    iteration_number = i + 1
    print("\n-----------------------------------------------------")
    print("DEBUG: Starting Iteration {} of {}".format(iteration_number, num_iterations))

    # --- Calculate current pressure ---
    print("DEBUG: Calculating current pressure...")
    current_pressure = initial_pressure + i * pressure_increment
    print("Current Pressure: {:.2f} Pa".format(current_pressure))

    # --- Modify pressure loads in MDB ---
    print("DEBUG: Updating pressure loads in the model...")
    try:
        for load_name in pressure_load_names:
            print("DEBUG: Accessing load: {}".format(load_name))
            load_object = myModel.loads[load_name]
            print("DEBUG: Setting magnitude for {} to {}".format(load_name, current_pressure))
            load_object.setValues(magnitude=current_pressure)
        print("DEBUG: Successfully updated {} pressure loads.".format(len(pressure_load_names)))
    except KeyError as e:
        print("ERROR: Pressure load name '{}' not found in model '{}'.".format(e, model_name))
        print("Please check the 'pressure_load_names' list and the CAE file.")
        sys.exit("Script aborted due to load name error.")
    except Exception as e:
        print("ERROR: Failed to update pressure loads.")
        print(e)
        sys.exit("Script aborted.")

    # --- Create and run the Abaqus job ---
    # Job 文件将在当前工作目录 (target_directory) 创建
    job_name = 'PressureStudy_Iter_{}'.format(iteration_number)
    print("DEBUG: Creating Abaqus job object: {}".format(job_name))
    try:
        # Create the job object
        myJob = mdb.Job(name=job_name, model=model_name,
                        description='Pressure study iteration {}'.format(iteration_number),
                        type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
                        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
                        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
                        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
                        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, numDomains=1, numGPUs=0)
        print("DEBUG: Job object created: {}".format(job_name))

        print("DEBUG: About to submit job: {}".format(job_name))
        myJob.submit(consistencyChecking=OFF)
        print("DEBUG: Job submission command issued for: {}".format(job_name))

        print("DEBUG: About to wait for job completion: {}".format(job_name))
        myJob.waitForCompletion()
        print("DEBUG: Job waitForCompletion finished for: {}".format(job_name))
        print("Job {} completed.".format(job_name)) # Original print message

    except AbaqusException as e:
        print("ERROR: Abaqus job creation, submission, or execution failed for {}.".format(job_name))
        print("Error message: {}".format(e))
        print("DEBUG: Aborting script due to job failure exception.")
        sys.exit("Script aborted due to job failure.")
    except Exception as e:
        print("ERROR: An unexpected error occurred during job handling for {}.".format(job_name))
        print(e)
        print("DEBUG: Aborting script due to unexpected job handling exception.")
        sys.exit("Script aborted.")

    # --- Post-processing: Extract results from ODB ---
    # ODB 文件在当前工作目录 (target_directory)
    odb_path = job_name + '.odb'
    print("DEBUG: Starting ODB post-processing for: {}".format(odb_path))
    odb = None
    max_values_for_frame = {var: 0.0 for var in output_variables}
    print("DEBUG: Initialized max_values_for_frame: {}".format(max_values_for_frame))

    try:
        print("DEBUG: Attempting to open ODB file: {}".format(odb_path))
        odb = openOdb(path=odb_path, readOnly=True) # openOdb 会在当前工作目录查找
        print("DEBUG: Successfully opened ODB file: {}".format(odb_path))

        # --- Get the Slave Surface region ---
        slave_surface_region = None
        print("DEBUG: Attempting to access slave surface: {} on instance: {}".format(contact_slave_surface_name, instance_name if instance_name else "Assembly"))
        try:
            if instance_name:
                print("DEBUG: Accessing instance: {}".format(instance_name))
                instance = odb.rootAssembly.instances[instance_name]
                print("DEBUG: Accessing surface on instance: {}".format(contact_slave_surface_name))
                slave_surface_region = instance.surfaces[contact_slave_surface_name]
            else:
                print("DEBUG: Accessing surface on assembly: {}".format(contact_slave_surface_name))
                slave_surface_region = odb.rootAssembly.surfaces[contact_slave_surface_name]
            print("DEBUG: Successfully accessed slave surface region.")
        except KeyError:
            print("ERROR: Could not find instance '{}' or surface '{}' in ODB '{}'.".format(instance_name, contact_slave_surface_name, odb_path))
            print("DEBUG: Skipping ODB data extraction for this iteration due to surface access error.")
            pass

        # --- Access the last frame of the specified step ---
        analysis_step = None
        last_frame = None
        if slave_surface_region:
            print("DEBUG: Attempting to access step: {}".format(step_name))
            try:
                analysis_step = odb.steps[step_name]
                print("DEBUG: Step '{}' accessed.".format(step_name))
                if analysis_step.frames:
                    print("DEBUG: Accessing last frame of step '{}'...".format(step_name))
                    last_frame = analysis_step.frames[-1]
                    print("DEBUG: Accessed last frame (Frame ID: {}) of step: {}".format(last_frame.frameId, step_name))
                else:
                    print("Warning: Step '{}' exists but contains no frames in ODB '{}'.".format(step_name, odb_path))
                    print("DEBUG: Setting last_frame to None as step has no frames.")
                    last_frame = None
            except KeyError:
                print("ERROR: Step '{}' not found in ODB '{}'.".format(step_name, odb_path))
                print("DEBUG: Setting last_frame to None as step was not found.")
                last_frame = None
            except IndexError:
                 print("Error accessing frames for step '{}'.".format(step_name))
                 print("DEBUG: Setting last_frame to None due to IndexError.")
                 last_frame = None

        # --- Extract Max Values for each variable ---
        if slave_surface_region and last_frame:
            print("DEBUG: Entering loop to extract max values for variables: {}".format(output_variables))
            for variable_name in output_variables:
                print("DEBUG: Processing variable: {}".format(variable_name))
                max_val = 0.0
                fieldOutput = None
                subset = None
                try:
                    print("DEBUG: Getting field output for '{}' from frame ID {}...".format(variable_name, last_frame.frameId))
                    fieldOutput = last_frame.fieldOutputs[variable_name]
                    print("DEBUG: Field output '{}' obtained.".format(variable_name))

                    print("DEBUG: Getting subset for '{}' on slave surface...".format(variable_name))
                    subset = fieldOutput.getSubset(region=slave_surface_region, position=NODAL)
                    print("DEBUG: Subset obtained for '{}'.".format(variable_name))

                    print("DEBUG: Checking bulkDataBlocks for subset...")
                    if subset.bulkDataBlocks:
                        print("DEBUG: Accessing data from bulkDataBlocks...")
                        data_array = np.array(subset.bulkDataBlocks[0].data).flatten()
                        if data_array.size > 0:
                            print("DEBUG: Calculating max value from bulkDataBlocks data...")
                            max_val = np.max(data_array)
                            print("DEBUG: Max value from bulkDataBlocks: {:.4e}".format(max_val))
                        else:
                            print("DEBUG: bulkDataBlocks data array is empty for '{}'.".format(variable_name))
                            max_val = 0.0
                    else:
                        print("DEBUG: bulkDataBlocks not found or empty. Checking subset.values for '{}'...".format(variable_name))
                        if subset.values:
                            try:
                                print("DEBUG: Iterating through subset.values to find max data...")
                                valid_data = [fv.data for fv in subset.values if isinstance(fv.data, (int, float))]
                                if valid_data:
                                     max_val = max(valid_data)
                                     print("DEBUG: Max value from subset.values iteration: {:.4e}".format(max_val))
                                else:
                                     print("DEBUG: No valid numeric data found in subset.values.")
                                     max_val = 0.0
                            except (AttributeError, TypeError) as fallback_e:
                                # <<< 修正: 恢复了之前的打印语句 >>>
                                print()
                                max_val = 0.0
                        else:
                            print("DEBUG: subset.values is also empty for '{}'.".format(variable_name))
                            max_val = 0.0

                    print("Max value for {}: {:.4e}".format(variable_name, max_val))

                except KeyError:
                    print("Warning: Field output '{}' not found in frame {}. Setting max to 0.0.".format(variable_name, last_frame.frameId))
                    max_val = 0.0
                except (AttributeError, IndexError, TypeError, ValueError) as e:
                    print("Warning: Error processing subset or calculating max for {}: {}. Setting max to 0.0.".format(variable_name, e))
                    max_val = 0.0
                except AbaqusException as e:
                     # <<< 修正: 恢复了之前的打印语句 >>>
                     print()
                     max_val = 0.0
                except odbAccess.OdbError as e: # <<< 已修正
                    print()
                    max_val = 0.0
                except Exception as e:
                    print("Warning: An unexpected error occurred processing {}: {}. Setting max to 0.0.".format(variable_name, e))
                    max_val = 0.0

                print("DEBUG: Storing max value {:.4e} for variable '{}'".format(max_val, variable_name))
                max_values_for_frame[variable_name] = max_val
            print("DEBUG: Finished extracting max values for all variables.")
        else:
            print("DEBUG: Skipping max value extraction because slave_surface_region or last_frame is invalid/not found.")

    except odbAccess.OdbError as e: # <<< 已修正
        print("ERROR: An OdbAccess error occurred opening or processing ODB '{}'.".format(odb_path))
        print(e)
        print("DEBUG: Skipping results extraction for this iteration due to OdbError.")
    except Exception as e:
        print("ERROR: An unexpected error occurred during ODB processing for '{}'.".format(odb_path))
        print(e)
        print("DEBUG: Skipping results extraction for this iteration due to unexpected exception.")
    finally:
        if odb:
            print("DEBUG: Attempting to close ODB file: {}".format(odb_path))
            odb.close()
            print("DEBUG: Closed ODB file: {}".format(odb_path))
        else:
             print("DEBUG: ODB was not opened or an error occurred before closing.")

    # --- Write results for the current iteration to Excel ---
    print("DEBUG: Preparing row data for Excel...")
    row_data = [iteration_number, current_pressure]
    for var in output_variables:
        row_data.append(max_values_for_frame.get(var, 0.0))

    print("DEBUG: Attempting to write row to Excel: {}".format(row_data))
    try:
        ws.append(row_data)
        print("DEBUG: Row successfully appended to Excel worksheet.")
    except Exception as e:
        print("ERROR: Failed to append data to Excel worksheet.")
        print(e)

    print("DEBUG: Iteration {} completed.".format(iteration_number))

# =============================================================================
# Script Finalization
# =============================================================================
print("\n-----------------------------------------------------")
print("DEBUG: Script execution finished main loop.")
# 保存 Excel 到指定的绝对路径
print("DEBUG: Attempting to save final Excel workbook to: {}".format(output_excel_file))
try:
    wb.save(filename=output_excel_file) # 使用包含路径的文件名
    print("\n=====================================================")
    print("Script finished successfully!")
    print("All {} iterations completed.".format(num_iterations))
    print("Results saved to: {}".format(os.path.abspath(output_excel_file)))
    print("=====================================================")
    print("DEBUG: Excel workbook saved successfully.")
except Exception as e:
    print("\nERROR: Failed to save the Excel file '{}'.".format(output_excel_file))
    print("Please ensure the file is not open in another application and you have write permissions.")
    print(e)
    print("DEBUG: Error occurred during final Excel save.")

# Optional: Close the MDB if desired
# print("DEBUG: Attempting to close MDB...")
# mdb.close()
# print("DEBUG: MDB closed.")

print("DEBUG: Script execution complete.")