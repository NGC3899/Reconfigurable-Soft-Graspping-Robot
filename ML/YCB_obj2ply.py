import trimesh
import os
import numpy as np

def convert_obj_to_ply_with_color(input_path, output_path, num_points):
    """
    将.obj网格文件转换为带RGB颜色的.ply点云文件。

    该函数会加载一个.obj文件及其关联的纹理或顶点颜色。然后，它从模型
    表面采样指定数量的点，并为每个点提取对应的RGB颜色。最后，将生成的
    彩色点云保存为.ply文件。

    参数:
    - input_path (str): 输入的.obj文件路径。
    - output_path (str): 输出的.ply文件路径。
    - num_points (int): 从网格表面采样的点的数量。
    """
    print(f"开始处理文件: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在于 '{input_path}'")
        return

    try:
        # 加载网格，trimesh会自动尝试加载纹理（.mtl 和 图像文件）
        # process=False 可以防止 trimesh 对模型进行不必要修改
        mesh = trimesh.load(input_path, process=False)
        
        # 对于由多个部分组成的模型，将其合并为一个单一的网格
        if isinstance(mesh, trimesh.Scene):
            print("检测到场景（Scene），正在合并所有几何体...")
            # 使用 `dump` 将场景中的所有几何体合并成一个网格
            mesh = mesh.dump(concatenate=True)

        if not isinstance(mesh, trimesh.Trimesh):
             print(f"错误：无法将 '{input_path}' 作为可用的网格加载。")
             return

        print(f"成功加载网格。顶点数: {len(mesh.vertices)}, 面数: {len(mesh.faces)}")

        # 从网格表面采样点，同时获取这些点所在的面（face）的索引
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        print(f"已从网格表面采样 {len(points)} 个点。")

        colors = None
        # 检查网格是否有纹理信息 (Texture)
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            print("检测到纹理信息，正在为采样点提取颜色...")
            try:
                # 获取纹理图像
                texture = mesh.visual.material.image
                if texture is None:
                    print("警告：找到了UV坐标，但无法加载纹理图像。")
                else:
                    # 计算采样点在三角面上的重心坐标
                    barycentric = trimesh.triangles.points_to_barycentric(
                        triangles=mesh.triangles[face_indices], points=points
                    )
                    
                    # 获取采样点所在面的顶点的UV坐标
                    face_uvs = mesh.visual.uv[mesh.faces[face_indices]]
                    
                    # 使用重心坐标插值计算每个采样点的UV坐标
                    interpolated_uv = (face_uvs * np.expand_dims(barycentric, axis=2)).sum(axis=1)
                    
                    # 从纹理图像中查找插值UV坐标对应的颜色
                    colors = trimesh.visual.color.uv_to_color(interpolated_uv, texture)
                    print("成功从纹理中提取颜色。")
            except Exception as e:
                print(f"从纹理提取颜色时出错: {e}")
                
        # 如果没有纹理，检查是否有顶点颜色 (Vertex Colors)
        elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
            print("检测到顶点颜色，正在为采样点插值颜色...")
            # 获取采样点所在面的顶点的颜色
            vertex_colors = mesh.visual.vertex_colors[mesh.faces[face_indices]]
            
            # 计算采样点在三角面上的重心坐标
            barycentric = trimesh.triangles.points_to_barycentric(
                triangles=mesh.triangles[face_indices], points=points
            )
            
            # 使用重心坐标对顶点颜色进行插值
            colors_float = (vertex_colors * np.expand_dims(barycentric, axis=2)).sum(axis=1)
            colors = colors_float.astype(np.uint8) # 转换为标准的颜色格式
            print("成功从顶点颜色中插值颜色。")

        else:
            print("警告：未在模型中找到纹理或顶点颜色信息。将生成不带颜色的点云。")

        # 创建点云对象，如果colors为None，则创建不带颜色的点云
        point_cloud = trimesh.points.PointCloud(points, colors=colors)

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        # 将点云导出为PLY文件
        point_cloud.export(file_obj=output_path, file_type='ply')
        
        print(f"彩色点云已成功保存到: {output_path}")

    except Exception as e:
        print(f"处理文件时发生严重错误: {e}")

# --- 主要执行部分 ---
if __name__ == '__main__':
    # --- 请在这里修改您的参数 ---
    
    # 1. 定义输入的 .obj 文件路径
    # 重要提示：确保.obj, .mtl, 和纹理图片文件都放在同一个目录下，
    # 并且.obj文件内部正确引用了.mtl文件。
    # 例如: "C:/YCB_Dataset/002_master_chef_can/google_16k/textured.obj"
    INPUT_FILE_PATH = r"C:\Users\admin\Desktop\YCB OBJ\models\models\025_mug\textured.obj" 
    
    # 2. 定义输出的 .ply 文件路径
    # 例如: "C:/PLY_Output/002_master_chef_can.ply"
    OUTPUT_FILE_PATH = r"C:\Users\admin\Desktop\YCB PLY\mug.ply"
    
    # 3. 定义需要采样的点的数量
    NUMBER_OF_POINTS = 4500 # GraspNet通常使用较多的点
    
    # --- 参数修改结束 ---
    
    print("--- 开始转换 (支持颜色) ---")
    convert_obj_to_ply_with_color(INPUT_FILE_PATH, OUTPUT_FILE_PATH, NUMBER_OF_POINTS)
    print("--- 转换完成 ---")
