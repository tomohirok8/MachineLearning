import torch



# Pytorch環境を確認
def check_pytorch():
    # pytorchでGPU使用可能か確認
    print('GPU available : ', torch.cuda.is_available())
    # Pytorchのバージョン確認
    print('Pytorch version : ', torch.__version__)
    # PyTorchで使用できるGPU（デバイス）数の確認
    print('GPU count : ', torch.cuda.device_count())
    # デフォルトのGPU番号（インデックス）取得
    print('GPU index : ', torch.cuda.current_device())
    # GPUの名称およびCUDA Compute Capability
    print('GPU device name : ', torch.cuda.get_device_name())
    print('GPU device capability : ', torch.cuda.get_device_capability())