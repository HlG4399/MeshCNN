from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run(epoch=-1):
    print('Running')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    model.classes = dataset.dataset.classes

    with open(model.save_dir + '/pred_classes.txt', 'w'):
        pass

    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run()
