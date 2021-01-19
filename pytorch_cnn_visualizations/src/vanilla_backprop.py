"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
import  numpy as np
from misc_functions import get_example_params


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        # print(self.model.CNN.conv[0])
        # exit()
        # first_layer = self.model.CNN.conv[0].filters[0]
        # Fconv1d
        # for i in range(10):
        #     print(f"{i} th")
        # for name, layer in self.model.CNN.conv[0]:
        #     print("name is ", name)
        #     print("layer is ", layer)
        # exit()
        first_layer = self.model.CNN.ln0
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.LongTensor(1, model_output.size()[-1]).zero_()
        # print(target_class)
        # exit()
        one_hot_output[0][target_class] = 1
        
        # one_hot_output
        # Backward pass
        # input_image.requires_grad = True
        # input_image.retain_grad()
        model_output.backward(gradient=one_hot_output)
        # cost = nn.NLLLoss()
        # loss = cost(pout, lab.long())
        # print(input_image.grad)
        # exit()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        # print(self.gradients.data.numpy().shape)
        # exit()
        gradients_as_arr = self.gradients.data.numpy()
        # gradients_as_arr = self.gradients.data.numpy()[0]
        gradients_as_arr = np.squeeze(gradients_as_arr)


        # print(gradients_as_arr.shape)
        tmp_m = np.mean(np.abs(gradients_as_arr),axis=0,keepdims=True)
        tmp_std = np.std(np.abs(gradients_as_arr),axis=0,keepdims=True)

        # print(tmp)
        # exit()


        # print(gradients_as_arr)

        gradients_as_arr = ( np.abs(gradients_as_arr) - tmp_m ) / tmp_std

        gradients_as_arr = gradients_as_arr * 1 + 1

        print("mean",np.mean(gradients_as_arr))

        gradients_as_arr = np.abs(gradients_as_arr)

        print("std",tmp_std)
        # print(gradients_as_arr)
        # print(np.max(gradients_as_arr))
        # exit()
        
        # normalize to 0 ~ 1
        # gradients_as_arr = np.abs(gradients_as_arr)

        # gradients_as_arr /= np.max(np.abs(gradients_as_arr),axis=2)
        # print(gradients_as_arr.shape)       
        
        return gradients_as_arr


# if __name__ == '__main__':
import copy
def vanilla(sincnet_model,input_data):
    # input_data [waveform,target]
    # Get params
    # sincnet = sincnet_model(CNN, DNN1, DNN2)
    # target_example = 1  # Snake
    # (original_image, prep_img, target_class, file_name_to_export) =\
    #     get_example_params(target_example)
    # Vanilla backprop
    VBP = VanillaBackprop(sincnet_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(copy.deepcopy(input_data[0]), input_data[1])
    # print("vanilla taken !!!!")
    
    # Save colored gradients
    # save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    # grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    # save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    print('Vanilla backprop completed')
    return vanilla_grads
