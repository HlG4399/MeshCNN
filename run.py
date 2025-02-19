from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import os
import shutil


def sync_empty_dirs(source_path, target_path, ignore_path=None):
    # 清空目标目录所有内容（保留目录本身）
    if os.path.exists(target_path):
        for item in os.listdir(target_path):
            item_path = os.path.join(target_path, item)
            if ignore_path is not None and ignore_path in item_path:
                item_test_path = item_path + '/test'
                if not os.path.exists(item_test_path):
                    os.makedirs(item_test_path)
                
                item_train_path = item_path + '/train'
                if not os.path.exists(item_train_path):
                    os.makedirs(item_train_path)
                
                continue

            if os.path.isfile(item_path):
                os.remove(item_path)  # 删除文件
            else:
                shutil.rmtree(item_path)  # 删除子目录
    else:
        os.makedirs(target_path)

    # 创建与源目录相同的子目录结构
    for root, dirs, files in os.walk(source_path):
        # 计算相对路径
        rel_path = os.path.relpath(root, source_path)
        if rel_path == ".":  # 根目录本身
            continue
        
        # 构建目标目录路径
        dest_dir = os.path.join(target_path, rel_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)


def sync_directories(source_dir, target_dir):
    """
    同步源目录和目标目录，使得目标目录中的文件与源目录保持一致。
    
    :param source_dir: 源目录路径
    :param target_dir: 目标目录路径
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取源目录中的所有文件名集合
    source_files = set()
    for filename in os.listdir(source_dir):
        full_source_path = os.path.join(source_dir, filename)
        if os.path.isfile(full_source_path):
            source_files.add(filename)
            
    # 获取目标目录中的所有文件名集合
    target_files = set()
    for filename in os.listdir(target_dir):
        full_target_path = os.path.join(target_dir, filename)
        if os.path.isfile(full_target_path):
            target_files.add(filename)
            
    # 找出需要新增的文件
    to_add = source_files - target_files
    
    # 找出需要删除的文件
    to_remove = target_files - source_files
    
    # 添加新文件
    for filename in to_add:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        shutil.copy2(source_path, target_path)
        print(f"已复制: {filename} → {target_path}")
        
    # 删除多余文件
    for filename in to_remove:
        target_path = os.path.join(target_dir, filename)
        os.remove(target_path)
        print(f"已删除: {filename} from {target_path}")


def copy_files_with_check(src_dir, max_copy=3):
    # 获取待处理文件列表
    all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    new_files = [f for f in all_files if '_copy' not in f and f.endswith('.obj')]

    # 执行复制操作
    if len(new_files) > 0:
        print(f"开始复制 {len(new_files)} 个新文件：")
        for file in new_files:
            src_dir = os.path.join(src_dir, file)
            src_dir_without_suffix = src_dir[:-4]
            for i in range(max_copy):
                shutil.copy2(src_dir, src_dir_without_suffix + f'_copy{i}.obj')
            print(f"√ 已复制 {file}")
    else:
        print("没有需要复制的新文件")


def run(epoch=-1):
    print('Running')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle

    source_folder = "./datasets/shrec_16"
    assert os.path.exists(source_folder)

    target_folder = opt.dataroot
    sync_empty_dirs(source_folder, target_folder, ignore_path='meshes_to_pred')

    alien_test_sub_folder = '/alien/test'
    alien_test_source_folder = source_folder + alien_test_sub_folder
    assert os.path.exists(alien_test_source_folder)

    alien_test_target_folder = target_folder + alien_test_sub_folder
    sync_directories(alien_test_source_folder, alien_test_target_folder)

    meshes_to_pred_folder = target_folder + '/meshes_to_pred/test'
    copy_files_with_check(meshes_to_pred_folder)

    dataset = DataLoader(opt)
    model = create_model(opt)
    model.classes = dataset.dataset.classes

    writer = Writer(opt)
    # run
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run()
