from options.test_options import TestOptions
from data import DataLoader
from models import create_model
import torch
import torch.onnx


def convert_pth_to_onnx(): 
	print('Running Converting')
	opt = TestOptions().parse()
	opt.serial_batches = True  # no shuffle
	DataLoader(opt)
	model = create_model(opt)
	model.load_network('latest')
	net = model.net
		
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net.to(device)
	net.eval()

	# Let's create a dummy input tensor
	mesh_index = 0
	dummy_input = (torch.randn(16, 5, 750, requires_grad=True).to(device), torch.tensor([mesh_index]).to(device))

	# Export the model   
	torch.onnx.export(
		net.module,                                         	# model being run 
		dummy_input,                                        	# model input (or a tuple for multiple inputs) 
		"mesh_classifier.onnx",                             	# where to save the model  
		export_params=True,                                 	# store the trained parameter weights inside the model file 
		opset_version=13,                                   	# the ONNX version to export the model to 
		do_constant_folding=True,                           	# whether to execute constant folding for optimization 
		input_names = [
      		'modelInput0',
        	'modelInputMesh'],                       			# the model's input names 
		output_names = ['modelOutput'],                     	# the model's output names 
		dynamic_axes={
      		'modelInput0' : {0: 'batch_size', 2: 'src_len'},   	# variable length axes
			'modelOutput' : {0 : 'batch_size'}})



if __name__ == '__main__':
    convert_pth_to_onnx()
