from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import os
import shutil


def sync_empty_dirs(source_path, target_path):
    # 清空目标目录所有内容（保留目录本身）
    if os.path.exists(target_path):
        for item in os.listdir(target_path):
            item_path = os.path.join(target_path, item)
            if 'meshes_to_pred' in item_path:
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


def run(epoch=-1):
    print('Running')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle

    source_folder = "./datasets/shrec_16"
    assert os.path.exists(source_folder)

    target_folder = opt.dataroot
    sync_empty_dirs(source_folder, target_folder)

    alien_test_sub_folder = '/alien/test'
    alien_test_source_folder = source_folder + alien_test_sub_folder
    assert os.path.exists(alien_test_source_folder)

    alien_test_target_folder = target_folder + alien_test_sub_folder
    sync_directories(alien_test_source_folder, alien_test_target_folder)

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
