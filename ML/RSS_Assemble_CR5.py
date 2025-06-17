from Dobot_Robot import Dobot_Robot
import rospy
from std_msgs.msg import String
import numpy as np
import time
import math
import socket
import ast

# 初始化IP，实例化机械臂类

tcp_host_ip = "192.168.5.1"

ESP32_ip = ""
ESP32_port = 8888
BUFFER_SIZE = 1024
CONNECTION_TIMEOUT = 10 # 连接超时时间（秒）
RECEIVE_TIMEOUT = 5    # 接收数据超时时间（秒）

# 全局变量
r = 0.0
n1 = 0.0
n2 = 0.0
n3 = 0.0
theta = 0.0

# 初始化ESP32配置
ESP32_IP = "192.168.159.132"  # <<< --- 修改这里为您的 ESP32 IP ---
TCP_PORT = 8888           # ESP32 服务器端口，与 ESP32 代码中设置的一致
BUFFER_SIZE = 1024        # 接收数据缓冲区大小
COMMAND_DELAY = 2         # 发送每条指令之间的延迟时间（秒）
ESP32_WIFI_SSID = "Mi 11 Pro" 

# 储存重构机构的初始位置对应MoveJ关节角度

Reconfig_mechanism_1 = [55.806*math.pi/180, 23.646*math.pi/180, 86.415*math.pi/180, -18.915*math.pi/180, -87.049*math.pi/180, -107.605*math.pi/180] # 机械臂末端圆心对准1号重构机构的关节角度
Reconfig_mechanism_2 = [55.806*math.pi/180, 23.646*math.pi/180, 86.415*math.pi/180, -18.915*math.pi/180, -87.049*math.pi/180, -107.605*math.pi/180] # 机械臂末端圆心对准2号重构机构的关节角度
Reconfig_mechanism_3 = [55.806*math.pi/180, 23.646*math.pi/180, 86.415*math.pi/180, -18.915*math.pi/180, -87.049*math.pi/180, -107.605*math.pi/180] # 机械臂末端圆心对准3号重构机构的关节角度

# 向ESJ32发送的命令


COMMAND_EM_ON ={
    1: "EM_1_ON",
    2: "EM_2_ON",
    3: "EM_3_ON",
}

COMMAND_EM_OFF ={
    1: "EM_1_OFF",
    2: "EM_2_OFF",
    3: "EM_3_OFF",
}

COMMAND_PPSV_ON ={
    1: "Valve_1_ON",
    2: "Valve_2_ON",
    3: "Valve_3_ON",
}

COMMAND_PPSV_OFF ={
    1: "Valve_1_OFF",
    2: "Valve_2_OFF",
    3: "Valve_3_OFF",
}

def ESP32_connection(command_str):

    response = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            print(f"正在连接到 ESP32 ({ESP32_ip}:{ESP32_port})...")
            s.settimeout(CONNECTION_TIMEOUT) # 设置连接超时
            s.connect((ESP32_ip, ESP32_port))
            print("连接成功！")

            # 发送命令 (确保编码为bytes)
            print(f"发送命令: '{command_str}'")
            s.sendall(command_str.encode('utf-8')) # 使用UTF-8编码

            # 等待并接收响应
            s.settimeout(RECEIVE_TIMEOUT) # 设置接收超时
            print("等待ESP32响应...")
            data_bytes = s.recv(BUFFER_SIZE)
            if data_bytes:
                response = data_bytes.decode('utf-8').strip() # 解码并去除首尾空白
                print(f"收到响应: '{response}'")
            else:
                print("未收到ESP32的响应或连接已关闭。")

        except socket.timeout as e:
            print(f"Socket超时错误: {e}")
        except ConnectionRefusedError:
            print(f"连接被拒绝。请确保ESP32服务器正在运行并且IP/端口正确。")
        except socket.error as e:
            print(f"Socket错误: {e}")
        except Exception as e:
            print(f"发生其他错误: {e}")
        finally:
            print("关闭连接。")
    return response


# 回调函数
def callback(msg):
    global r, n1, n2, n3, theta # 更新全局变量列表

    data_str = msg.data # PEP8建议不要用str作为变量名，因为它会覆盖内置的str类型
    rospy.loginfo("Received data: %s", data_str)

    # 1. 按照分号将消息分割
    parts = data_str.strip().split(';')
    parsed_values_temp = {} # 使用临时字典存储解析的中间值

    for part in parts:
        part = part.strip()
        if not part: # 跳过因末尾';'产生的空字符串
            continue

        # 2. 按 ':' 分割键和值
        key_value = part.split(':', 1) # 只在第一个冒号处分割
        if len(key_value) == 2:
            key = key_value[0].strip()
            value_str = key_value[1].strip()

            try:
                if key == "number_of_slot":
                    # 特殊处理 number_of_slot，它现在是一个向量字符串，如 "[0, 2, 4]"
                    # 使用 ast.literal_eval 将其安全转换为列表
                    vector_list = ast.literal_eval(value_str)
                    if isinstance(vector_list, list) and len(vector_list) == 3:
                        # 确保列表中的每个元素都可以转换为浮点数
                        parsed_values_temp[key] = [float(item) for item in vector_list]
                    else:
                        rospy.logwarn(f"'{key}' 的值 '{value_str}' 不是一个包含3个元素的有效向量格式。")
                        parsed_values_temp[key] = None # 标记解析失败
                elif key == "radius" or key == "step_angle":
                    # 其他值仍然是单个浮点数
                    parsed_values_temp[key] = float(value_str)
                else:
                    rospy.logwarn(f"未知的键: '{key}'")

            except ValueError: # 包括 float() 转换失败和 ast.literal_eval 对非数字列表元素的float转换失败
                rospy.logwarn(f"无法将键 '{key}' 的值 '{value_str}' 转换为期望的数字类型。")
                parsed_values_temp[key] = None # 标记解析失败
            except SyntaxError: # ast.literal_eval 无法解析字符串
                rospy.logwarn(f"无法将键 '{key}' 的值 '{value_str}' 解析为列表格式。")
                parsed_values_temp[key] = None # 标记解析失败
        else:
            rospy.logwarn(f"格式错误的部分 (缺少冒号或格式无效): '{part}'")


    # 3. 分配参数给全局变量
    # 处理 r
    if "radius" in parsed_values_temp and parsed_values_temp["radius"] is not None:
        r = parsed_values_temp["radius"]
    else:
        rospy.logwarn("消息中未找到或无法解析 'radius'")
        r = 0.0 # 或者保持上一个值，或抛出错误

    # 处理 n1, n2, n3
    if "number_of_slot" in parsed_values_temp and parsed_values_temp["number_of_slot"] is not None:
        n_vector = parsed_values_temp["number_of_slot"]
        # 再次检查是否是包含3个浮点数的列表 (尽管在上面已经检查过，这里作为安全措施)
        if isinstance(n_vector, list) and len(n_vector) == 3 and all(isinstance(item, float) for item in n_vector):
            n1 = n_vector[0]
            n2 = n_vector[1]
            n3 = n_vector[2]
        else:
            rospy.logwarn(f"'number_of_slot' 解析结果不是包含3个浮点数的列表: {n_vector}")
            n1, n2, n3 = 0.0, 0.0, 0.0 # 设置默认值
    else:
        rospy.logwarn("消息中未找到或无法解析 'number_of_slot'")
        n1, n2, n3 = 0.0, 0.0, 0.0 # 设置默认值
            
    # 处理 theta
    if "step_angle" in parsed_values_temp and parsed_values_temp["step_angle"] is not None:
        theta = parsed_values_temp["step_angle"]
    else:
        rospy.logwarn("消息中未找到或无法解析 'step_angle'")
        theta = 0.0 # 或者保持上一个值

    rospy.loginfo(f"解析结果: r={r}, n1={n1}, n2={n2}, n3={n3}, theta={theta}")



def Assemble_Motion_Planning(r, n, theta, mecha_num, robot = Dobot_Robot(tcp_host_ip)):

    # 第一步将机械臂末端圆心对准对应的重构位置
    robot.move_j(mecha_num)
    time.sleep(0.5)

    # 第二步获得当前位置(position+rpy)并将模块对准所求半径r的圆弧位置
    current_pose = robot.get_current_pose()
    current_pose[0] += r # 假定沿x轴正方向运动
    robot.move_l(current_pose)
    time.sleep(0.5)

    # 第三步旋转机械臂末端关节至对应位置，angle = n*theta
    angle = [0*math.pi/180, 0*math.pi/180, 0*math.pi/180, 0*math.pi/180, 0*math.pi/180, n*theta*math.pi/180]
    robot.move_j(angle)
    time.sleep(0.5)

    # 第四步提前为电磁铁上电，确保拼装时手指模块不会晃动
    ESP32_connection(COMMAND_EM_ON[mecha_num])
    time.sleep(1)

    # 第五步为推拉式电磁阀上电，压紧手指模块确保其拼装时不晃动
    ESP32_connection(COMMAND_EM_OFF[mecha_num])
    time.sleep(1)

    # 第六步对准位置后下降
    z_down = 0.15 # 该值为机械臂下降拼装模块的距离
    current_pose = robot.get_current_pose()
    current_pose[2] -= z_down
    robot.move_l(current_pose)
    time.sleep(1)

    # 第七步为推拉式电磁阀下电
    ESP32_connection(COMMAND_PPSV_OFF[mecha_num])
    time.sleep(1)

    # 第八步为电磁铁下电
    ESP32_connection(COMMAND_EM_OFF[mecha_num])
    time.sleep(1)

    # 第九步抬升机械臂，完成拼装动作
    current_pose = robot.get_current_pose()
    current_pose[2] += z_down
    robot.move_l(current_pose)
    time.sleep(1)

    rospy.loginfo("完成拼装过程")


def Disassemble_Motion_Planning(r, n, theta, mecha_num, robot = Dobot_Robot(tcp_host_ip)):

    # 前三步与assemble的过程一致，由于拼装过程第一步自动回归原位置因此手指的坐标相对于该点保持不变，无需改变对准过程

    # 第一步将机械臂末端圆心对准对应的重构位置
    robot.move_j(mecha_num)
    time.sleep(0.5)

    # 第二步获得当前位置(position+rpy)并将模块对准所求半径r的圆弧位置
    current_pose = robot.get_current_pose()
    current_pose[0] += r # 假定沿x轴正方向运动
    robot.move_l(current_pose)
    time.sleep(0.5)

    # 第三步旋转机械臂末端关节至对应位置，angle = n*theta
    angle = [0*math.pi/180, 0*math.pi/180, 0*math.pi/180, 0*math.pi/180, 0*math.pi/180, n*theta*math.pi/180]
    robot.move_j(angle)
    time.sleep(0.5)

    # 第四步打开电磁铁，确保下降过程中手指模块对齐
    ESP32_connection(COMMAND_EM_ON[mecha_num])
    time.sleep(1)

    # 第五步对准位置后下降
    z_down = 0.15 # 该值为机械臂下降拼装模块的距离
    current_pose = robot.get_current_pose()
    current_pose[2] -= z_down
    robot.move_l(current_pose)
    time.sleep(1)

    # 第六步打开推拉式电磁阀，加紧手指确保拆卸成功
    ESP32_connection(COMMAND_PPSV_ON[mecha_num])
    time.sleep(1)

    # 第七步抬升机械臂，拆卸手指模块
    current_pose = robot.get_current_pose()
    current_pose[2] += z_down
    robot.move_l(current_pose)
    time.sleep(1)

    # 第八步关闭推拉式电磁阀
    ESP32_connection(COMMAND_PPSV_OFF[mecha_num])
    time.sleep(1)

    # 第九步关闭电磁铁，完成拆卸过程
    ESP32_connection(COMMAND_EM_OFF[mecha_num])
    time.sleep(1)

    rospy.loginfo("完成拆卸过程")

if __name__ == '__main__':

    # 这一步已经初始化了ros节点
    robot = DOBOT_Robot(tcp_host_ip)

    # 该订阅者订阅来自构型优化程序发布的构型
    rospy.Subscriber('/Opt_Config', String, callback)
