# -*- coding: utf-8 -*-

# =============================================================================
# Abaqus Python Script for Parametric Pressure Study
# Extracts MAX CPRESS, CSHEAR1, CSHEAR2 from a Contact Slave Surface
# Includes Debug Print Statements
# =============================================================================
#
# Description:
# This script automates Abaqus simulations by incrementally increasing pressure.
# For each run, it extracts the maximum values of specified contact stress
# variables (CPRESS, CSHEAR1, CSHEAR2) from a defined slave surface in the ODB
# and logs results (iteration, pressure, max values) into an Excel file.
#
# Author: [Your Name/Organization] - Modified based on user request
# Date: [Date]
# Version: 2.1 (with Debugging)
#
# Prerequisites:
# 1. A working Abaqus.cae model file.
# 2. The 'openpyxl' and 'numpy' Python libraries must be installed in the
#    Abaqus Python environment.
#    (Run: abaqus python -m pip install openpyxl numpy)
# 3. Field outputs CPRESS, CSHEAR1, CSHEAR2 must be requested for the
#    specified step and associated with the contact interaction.
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

# Try to import external libraries and provide guidance if they fail
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

import os # For path manipulation
import sys # For exiting gracefully
print("DEBUG: Standard Python modules imported.")

# =============================================================================
# USER CONFIGURABLE PARAMETERS - MODIFY THESE VALUES
# =============================================================================
print("DEBUG: Setting user configurable parameters...")
# --- File and Model Identification ---
cae_file_path = 'D:\softgripper2.cae' # <<< 修改: 完整.cae 文件路径
model_name = 'Model-1'                     # <<< 修改:.cae 文件中的模型名称

# --- Load Control ---
# 需要在每次迭代中修改其大小的所有压力载荷的名称列表
pressure_load_names = [
    'pressure1', 'pressure2', 'pressure3',
    'pressure4a', 'pressure4b', 'pressure4c',
    'pressure4d', 'pressure4e', 'pressure4f',
    'pressure5a', 'pressure5b', 'pressure5c',
    'pressure5d', 'pressure5e'
] # <<< 修改: 包含所有14个压力载荷的准确名称
num_iterations = 10                        # 总仿真迭代次数
initial_pressure = 10000.0                 # 初始压力值 (Pa)
pressure_increment = 1000.0                # 每次迭代增加的压力值 (Pa)

# --- Output Configuration ---
output_excel_file = 'Max_Contact_Stress_Results.xlsx' # <<< 修改: 输出 Excel 文件名
# --- ODB Result Extraction Parameters ---
# 包含从属表面的部件实例名称 (如果表面定义在实例上)
# 如果表面直接定义在装配上，请将此设置为空字符串 '' 或 None
instance_name = 'PART-SOFT-FINGER-1'       # <<< 修改: 包含从属表面的实例名称
# 需要提取数据的接触从属表面 (Slave Surface) 的名称
contact_slave_surface_name = 's_Surf-15'   # <<< 修改: 接触相互作用 'touch' 中的从属表面名称
# 需要提取结果的分析步名称
step_name = 'Bending'                 # <<< 修改: 分析步名称
# 需要提取最大值的场输出变量列表 (确保这些名称在 ODB 中存在!)
output_variables = ["CPRESS", "CSHEAR1", "CSHEAR2"] # <<< 修正: 使用字符串列表
print("DEBUG: User parameters set.")

# =============================================================================
# Script Initialization
# =============================================================================
print("Starting Abaqus parametric pressure study script...")
print("Output will be saved to: {}".format(os.path.abspath(output_excel_file)))

# --- Initialize Excel Workbook ---
print("DEBUG: Initializing Excel workbook...")
# Define headers based on output_variables
excel_headers = ["Iteration", "Pressure"] + ["Max " + var for var in output_variables] # <<< 修正: 添加固定表头并使用列表连接

# Check if file exists to avoid overwriting header accidentally if appending later
# For simplicity, this version overwrites the file if it exists.
# If you need to append, load the workbook instead.
wb = Workbook()
ws = wb.active
ws.title = 'Max Contact Data'
print("DEBUG: Appending headers to Excel sheet...")
ws.append(excel_headers)
print("DEBUG: Excel headers written: {}".format(excel_headers))

# --- Open the CAE file and access the model ---
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
                        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, numDomains=1, numGPUs=0) # Adjust job settings as needed
        print("DEBUG: Job object created: {}".format(job_name))

        print("DEBUG: About to submit job: {}".format(job_name))
        # Submit the job for analysis
        myJob.submit(consistencyChecking=OFF) # Set consistencyChecking=ON for more checks if needed
        print("DEBUG: Job submission command issued for: {}".format(job_name))

        print("DEBUG: About to wait for job completion: {}".format(job_name))
        # Wait for the analysis to complete before proceeding
        myJob.waitForCompletion()
        print("DEBUG: Job waitForCompletion finished for: {}".format(job_name))
        print("Job {} completed.".format(job_name)) # Original print message

    except AbaqusException as e:
        print("ERROR: Abaqus job creation, submission, or execution failed for {}.".format(job_name))
        print("Error message: {}".format(e))
        # Decide whether to continue to next iteration or stop
        print("DEBUG: Aborting script due to job failure exception.")
        sys.exit("Script aborted due to job failure.") # Added exit here for clarity
    except Exception as e:
        print("ERROR: An unexpected error occurred during job handling for {}.".format(job_name))
        print(e)
        print("DEBUG: Aborting script due to unexpected job handling exception.")
        sys.exit("Script aborted.")

    # --- Post-processing: Extract results from ODB ---
    odb_path = job_name + '.odb'
    print("DEBUG: Starting ODB post-processing for: {}".format(odb_path))
    odb = None # Initialize odb to None for finally block
    # Initialize max values dictionary with default 0.0
    max_values_for_frame = {var: 0.0 for var in output_variables}
    print("DEBUG: Initialized max_values_for_frame: {}".format(max_values_for_frame))

    try:
        # Open the output database in read-only mode
        print("DEBUG: Attempting to open ODB file: {}".format(odb_path))
        odb = openOdb(path=odb_path, readOnly=True)
        print("DEBUG: Successfully opened ODB file: {}".format(odb_path))

        # --- Get the Slave Surface region ---
        slave_surface_region = None
        print("DEBUG: Attempting to access slave surface: {} on instance: {}".format(contact_slave_surface_name, instance_name if instance_name else "Assembly"))
        try:
            if instance_name: # Surface defined on an instance
                print("DEBUG: Accessing instance: {}".format(instance_name))
                instance = odb.rootAssembly.instances[instance_name]
                print("DEBUG: Accessing surface on instance: {}".format(contact_slave_surface_name))
                slave_surface_region = instance.surfaces[contact_slave_surface_name]
            else: # Surface defined directly on the assembly
                print("DEBUG: Accessing surface on assembly: {}".format(contact_slave_surface_name))
                slave_surface_region = odb.rootAssembly.surfaces[contact_slave_surface_name]
            print("DEBUG: Successfully accessed slave surface region.")
        except KeyError:
            print("ERROR: Could not find instance '{}' or surface '{}' in ODB '{}'.".format(instance_name, contact_slave_surface_name, odb_path))
            print("DEBUG: Skipping ODB data extraction for this iteration due to surface access error.")
            # Allow writing default 0.0 values later by letting the loop continue
            pass # Continue to finally block after this try

        # --- Access the last frame of the specified step ---
        analysis_step = None
        last_frame = None
        if slave_surface_region: # Only proceed if surface was found
            print("DEBUG: Attempting to access step: {}".format(step_name))
            try:
                analysis_step = odb.steps[step_name]
                print("DEBUG: Step '{}' accessed.".format(step_name))
                if analysis_step.frames:
                    print("DEBUG: Accessing last frame of step '{}'...".format(step_name))
                    last_frame = analysis_step.frames[-1] # Get the last frame
                    print("DEBUG: Accessed last frame (Frame ID: {}) of step: {}".format(last_frame.frameId, step_name))
                else:
                    print("Warning: Step '{}' exists but contains no frames in ODB '{}'.".format(step_name, odb_path))
                    print("DEBUG: Setting last_frame to None as step has no frames.")
                    last_frame = None # Ensure last_frame is None
            except KeyError:
                print("ERROR: Step '{}' not found in ODB '{}'.".format(step_name, odb_path))
                print("DEBUG: Setting last_frame to None as step was not found.")
                last_frame = None # Ensure last_frame is None if step not found
            except IndexError: # Should be caught by checking analysis_step.frames
                 print("Error accessing frames for step '{}'.".format(step_name))
                 print("DEBUG: Setting last_frame to None due to IndexError.")
                 last_frame = None

        # --- Extract Max Values for each variable ---
        if slave_surface_region and last_frame:
            print("DEBUG: Entering loop to extract max values for variables: {}".format(output_variables))
            for variable_name in output_variables:
                print("DEBUG: Processing variable: {}".format(variable_name))
                max_val = 0.0 # Reset max_val for each variable
                fieldOutput = None
                subset = None
                try:
                    # Get the field output object for the current variable
                    print("DEBUG: Getting field output for '{}' from frame ID {}...".format(variable_name, last_frame.frameId))
                    fieldOutput = last_frame.fieldOutputs[variable_name]
                    print("DEBUG: Field output '{}' obtained.".format(variable_name))

                    # Get the subset of the field output on the slave surface nodes
                    print("DEBUG: Getting subset for '{}' on slave surface...".format(variable_name))
                    subset = fieldOutput.getSubset(region=slave_surface_region, position=NODAL) # Assuming NODAL output for contact
                    print("DEBUG: Subset obtained for '{}'.".format(variable_name))

                    # Use bulkDataBlocks for efficient max calculation
                    print("DEBUG: Checking bulkDataBlocks for subset...")
                    if subset.bulkDataBlocks:
                        # Assuming data is in the first block for nodal data on a single surface
                        print("DEBUG: Accessing data from bulkDataBlocks...")
                        # Ensure data_array exists and is numpy array for np.max
                        data_array = np.array(subset.bulkDataBlocks[0].data).flatten() # Get data from first block, ensure numpy array
                        if data_array.size > 0: # Check if array is not empty
                            print("DEBUG: Calculating max value from bulkDataBlocks data...")
                            max_val = np.max(data_array)
                            print("DEBUG: Max value from bulkDataBlocks: {:.4e}".format(max_val))
                        else:
                            print("DEBUG: bulkDataBlocks data array is empty for '{}'.".format(variable_name))
                            max_val = 0.0 # Ensure max_val is 0.0 if empty
                    else:
                        # Fallback: Check if.values exist (less efficient)
                        print("DEBUG: bulkDataBlocks not found or empty. Checking subset.values for '{}'...".format(variable_name))
                        if subset.values:
                            try:
                                # Assuming scalar data is in.data attribute
                                print("DEBUG: Iterating through subset.values to find max data...")
                                valid_data = [fv.data for fv in subset.values if isinstance(fv.data, (int, float))]
                                if valid_data:
                                     max_val = max(valid_data)
                                     print("DEBUG: Max value from subset.values iteration: {:.4e}".format(max_val))
                                else:
                                     print("DEBUG: No valid numeric data found in subset.values.")
                                     max_val = 0.0 # Ensure 0.0 if no valid data
                            except (AttributeError, TypeError) as fallback_e:
                                print()
                                max_val = 0.0 # Default on error
                        else:
                            print("DEBUG: subset.values is also empty for '{}'.".format(variable_name))
                            max_val = 0.0 # Ensure 0.0 if no values

                    # Original print message retained
                    print("Max value for {}: {:.4e}".format(variable_name, max_val))

                except KeyError:
                    print("Warning: Field output '{}' not found in frame {}. Setting max to 0.0.".format(variable_name, last_frame.frameId))
                    max_val = 0.0 # Ensure default value
                except (AttributeError, IndexError, TypeError, ValueError) as e:
                    print("Warning: Error processing subset or calculating max for {}: {}. Setting max to 0.0.".format(variable_name, e))
                    max_val = 0.0 # Ensure default value
                except AbaqusException as e:
                     print()
                     max_val = 0.0 # Ensure default value
                except Exception as e:
                    print("Warning: An unexpected error occurred processing {}: {}. Setting max to 0.0.".format(variable_name, e))
                    max_val = 0.0 # Ensure default value

                # Store the calculated (or default 0.0) max value
                print("DEBUG: Storing max value {:.4e} for variable '{}'".format(max_val, variable_name))
                max_values_for_frame[variable_name] = max_val
            print("DEBUG: Finished extracting max values for all variables.")
        else:
            print("DEBUG: Skipping max value extraction because slave_surface_region or last_frame is invalid/not found.")


    except odbAccess.OdbError as e:  # <<< 修改: 使用 odbAccess.OdbError
        print("ERROR: An OdbAccess error occurred opening or processing ODB '{}'.".format(odb_path))
        print(e)
        print("DEBUG: Skipping results extraction for this iteration due to OdbError.")
    except Exception as e:
        print("ERROR: An unexpected error occurred during ODB processing for '{}'.".format(odb_path))
        print(e)
        print("DEBUG: Skipping results extraction for this iteration due to unexpected exception.")
    finally:
        # Ensure the ODB is closed even if errors occurred
        if odb:
            print("DEBUG: Attempting to close ODB file: {}".format(odb_path))
            odb.close()
            print("DEBUG: Closed ODB file: {}".format(odb_path))
        else:
             print("DEBUG: ODB was not opened or an error occurred before closing.")

    # --- Write results for the current iteration to Excel ---
    print("DEBUG: Preparing row data for Excel...")
    # Prepare row data in the order defined by excel_headers
    row_data = [iteration_number, current_pressure]
    for var in output_variables:
        # Use.get with default 0.0 for safety, though we initialized the dict earlier
        row_data.append(max_values_for_frame.get(var, 0.0))

    print("DEBUG: Attempting to write row to Excel: {}".format(row_data))
    try:
        ws.append(row_data)
        print("DEBUG: Row successfully appended to Excel worksheet.")
    except Exception as e:
        print("ERROR: Failed to append data to Excel worksheet.")
        print(e)
        # Consider saving what we have so far or stopping

    print("DEBUG: Iteration {} completed.".format(iteration_number))
    # print("-----------------------------------------------------") # Moved newline to loop start


# =============================================================================
# Script Finalization
# =============================================================================
print("\n-----------------------------------------------------")
print("DEBUG: Script execution finished main loop.")
print("DEBUG: Attempting to save final Excel workbook to: {}".format(output_excel_file))
try:
    wb.save(filename=output_excel_file)
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

# Optional: Close the MDB if desired (may not be necessary if script exits)
# print("DEBUG: Attempting to close MDB...")
# mdb.close()
# print("DEBUG: MDB closed.")

print("DEBUG: Script execution complete.")