import trimesh
import os
import numpy as np

def convert_obj_to_ply(input_path, output_path, num_points, uniform_color=None):
    """
    将.obj网格文件转换为.ply点云文件，可以选择性地应用统一的颜色。

    该函数会加载一个.obj文件。如果提供了 `uniform_color` 参数，所有采样点都
    将被赋予该颜色。否则，函数会尝试从模型的纹理或顶点颜色中提取颜色。
    最后，将生成的点云保存为.ply文件。

    参数:
    - input_path (str): 输入的.obj文件路径。
    - output_path (str): 输出的.ply文件路径。
    - num_points (int): 从网格表面采样的点的数量。
    - uniform_color (list or tuple, optional): 一个包含3个元素的RGB颜色列表，
      数值范围为 0-255。例如：[30, 120, 181]。如果提供此参数，将覆盖
      模型原有的颜色。默认为 None。
    """
    print(f"开始处理文件: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在于 '{input_path}'")
        return

    try:
        # 加载网格，process=False 可以防止 trimesh 对模型进行不必要修改
        mesh = trimesh.load(input_path, process=False)
        
        # 对于由多个部分组成的模型，将其合并为一个单一的网格
        if isinstance(mesh, trimesh.Scene):
            print("检测到场景（Scene），正在合并所有几何体...")
            mesh = mesh.dump(concatenate=True)

        if not isinstance(mesh, trimesh.Trimesh):
             print(f"错误：无法将 '{input_path}' 作为可用的网格加载。")
             return

        print(f"成功加载网格。顶点数: {len(mesh.vertices)}, 面数: {len(mesh.faces)}")

        # 从网格表面采样点，同时获取这些点所在的面（face）的索引
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        print(f"已从网格表面采样 {len(points)} 个点。")

        colors = None
        # --- 颜色处理逻辑 ---
        # 1. 优先使用用户指定的统一颜色
        if uniform_color is not None:
            print(f"正在为所有点应用统一颜色: {uniform_color}")
            if not (isinstance(uniform_color, (list, tuple)) and len(uniform_color) == 3):
                print("错误：'uniform_color' 必须是一个包含3个整数元素的列表或元组 (R, G, B)。")
                return
            
            # 创建一个颜色数组，其中每一行的值都是指定的颜色 (uint8格式)
            color_array = np.array(uniform_color, dtype=np.uint8)
            colors = np.tile(color_array, (len(points), 1))
            print("成功应用统一颜色。")
        
        # 2. 如果没有指定统一颜色，则尝试从模型文件中提取
        else:
            print("未指定统一颜色，尝试从模型文件提取颜色...")
            # 检查网格是否有纹理信息 (Texture)
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
                print("检测到纹理信息，正在为采样点提取颜色...")
                try:
                    texture = mesh.visual.material.image
                    if texture is None:
                        print("警告：找到了UV坐标，但无法加载纹理图像。")
                    else:
                        barycentric = trimesh.triangles.points_to_barycentric(
                            triangles=mesh.triangles[face_indices], points=points
                        )
                        face_uvs = mesh.visual.uv[mesh.faces[face_indices]]
                        interpolated_uv = (face_uvs * np.expand_dims(barycentric, axis=2)).sum(axis=1)
                        colors = trimesh.visual.color.uv_to_color(interpolated_uv, texture)
                        print("成功从纹理中提取颜色。")
                except Exception as e:
                    print(f"从纹理提取颜色时出错: {e}")
                    
            # 如果没有纹理，检查是否有顶点颜色 (Vertex Colors)
            elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
                print("检测到顶点颜色，正在为采样点插值颜色...")
                barycentric = trimesh.triangles.points_to_barycentric(
                    triangles=mesh.triangles[face_indices], points=points
                )
                vertex_colors = mesh.visual.vertex_colors[mesh.faces[face_indices]]
                colors_float = (vertex_colors * np.expand_dims(barycentric, axis=2)).sum(axis=1)
                colors = colors_float.astype(np.uint8)
                print("成功从顶点颜色中插值颜色。")
            else:
                print("警告：未在模型中找到纹理或顶点颜色信息。将生成不带颜色的点云。")

        # 创建点云对象
        point_cloud = trimesh.points.PointCloud(points, colors=colors)

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        # 将点云导出为PLY文件
        point_cloud.export(file_obj=output_path, file_type='ply')
        
        print(f"点云已成功保存到: {output_path}")

    except Exception as e:
        print(f"处理文件时发生严重错误: {e}")

# --- 主要执行部分 ---
if __name__ == '__main__':
    # --- 请在这里修改您的参数 ---
    
    # 1. 定义输入的 .obj 文件路径
    INPUT_FILE_PATH = r"C:\Users\admin\Desktop\stanford bunny\stanford-bunny\source\bunny.obj"
    
    # 2. 定义输出的 .ply 文件路径
    OUTPUT_FILE_PATH = r"C:\Users\admin\Desktop\stanford bunny\stanford-bunny\source\bunny_colored.ply"
    
    # 3. 定义需要采样的点的数量
    NUMBER_OF_POINTS = 4500 
    
    # 4. 定义统一的颜色 (RGB, 0-255)
    # 这是与您之前“精美可视化脚本”中一致的专业蓝色
    # 如果您想恢复使用模型自带的纹理颜色，请将此行设置为 UNIFORM_COLOR = None
    UNIFORM_COLOR = [30, 120, 181] # Professional Blue

    # --- 参数修改结束 ---
    
    print("--- 开始转换 ---")
    convert_obj_to_ply(INPUT_FILE_PATH, OUTPUT_FILE_PATH, NUMBER_OF_POINTS, uniform_color=UNIFORM_COLOR)
    print("--- 转换完成 ---")
