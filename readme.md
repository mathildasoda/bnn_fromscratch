# BNN by hand



code can be run by using `python [name].py`

weights generated and saved under `weights_[nn or bnn]_[data name].pkl`

Training and validation plot saved under `[nn or bnn]_[data_name].png`

----------------------------------------------------------------------
MNIST architecture: FC, ReLu, FC, Softmax, Cross entropy
Boston housing architecture: FC, ReLu, FC, Linear, SE

-----------------------------------------------------------------------

Miscellaneous results:

+ running `nn_mnist.py`:
	Final RMSE: 84.97591925123854
	Final training accuracy: 0.5515515515515516
	Final validation accuracy: 0.5252525252525253
	Weights and biases are saved as weights_nn_mnist.pkl
	Training vs Validation plot: nn_mnist.png
	With adam
	Time took: ~ 3 minutes

	>> 100 more epoch:
	Final RMSE: 80.49518282056779
	Final training accuracy: 0.5975975975975976
	Final validation accuracy: 0.5757575757575758
	Time took: few seconds


+ running `bnn_mnist.py`
	Final training accuracy: 0.08708708708708708
	Final validation accuracy: 0.0707070707070707
	Time took: ~ 2.5 hours

	>> 100 more epoch after pretrained using nn_mnist weights + biases:
	Final training accuracy: 0.5425425425425425
	Final validation accuracy: 0.5252525252525253

	Time took: ~12 minutes


+ running `nn_boston.py`
	Kept returning nan -> showing another good side of bnn

+ running `bnn_boston.py`
	Time took: ~ 3 minutes
