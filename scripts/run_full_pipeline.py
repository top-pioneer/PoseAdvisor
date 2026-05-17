import subprocess
import os
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_full_pipeline():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_dir = os.path.join(project_root, "demo", "input")
    output_root = os.path.join(project_root, "outputs")
    infer_pose_path = os.path.join(project_root, "infer_pose.py")
    fit_smplx_path = os.path.join(project_root, "fit_smplx_to_pose75.py")
    checkpoint_path = os.path.join(project_root, "model", "latest.pth")
    
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"在 {input_dir} 中没有找到图片文件！")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始处理...")
    
    for img_path in image_files:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n正在处理: {file_name}")
        
        pose_out_dir = os.path.join(output_root, file_name, "poses")
        smplx_out_dir = os.path.join(output_root, file_name, "smplx")
        
        os.makedirs(pose_out_dir, exist_ok=True)
        os.makedirs(smplx_out_dir, exist_ok=True)
        
        infer_cmd = [
            "python", infer_pose_path,
            "--mode", "generate",
            "--checkpoint", checkpoint_path,
            "--image", img_path,
            "--num_samples", "8",
            "--out_dir", pose_out_dir
        ]
        
        print(f"  开始执行第一步：生成姿态...")
        result1 = subprocess.run(infer_cmd, check=True)
        if result1.returncode != 0:
            print(f"  {file_name} 第一步执行失败！跳过该图片")
            continue
        
        npy_path = os.path.join(pose_out_dir, f"{file_name}_generated.npy")
        for index in range(8):
            output_obj = os.path.join(smplx_out_dir, f"{file_name}_fit_{index}.obj")
            
            fit_cmd = [
                "python", fit_smplx_path,
                "--auto_axes",
                "--npy", npy_path,
                "--index", str(index),
                "--steps", "700",
                "--lr", "0.01",
                "--stage1_ratio", "0.55",
                "--pose_reg", "0.03",
                "-o", output_obj
            ]
            
            print(f"  开始执行第二步：拟合第 {index+1}/8 个姿态...")
            result2 = subprocess.run(fit_cmd, check=True)
            if result2.returncode != 0:
                print(f"  {file_name} 第 {index} 个姿态拟合失败！继续下一个")
                continue
        
        print(f"  {file_name} 处理完成！")
    
    print("\n所有图片处理完成！")

if __name__ == "__main__":
    run_full_pipeline()
