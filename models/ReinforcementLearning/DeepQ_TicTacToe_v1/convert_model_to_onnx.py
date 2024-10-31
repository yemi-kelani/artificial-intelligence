import onnx
import torch
import argparse
import onnxruntime as ort
from DeepQAgent import DeepQAgentOnnxVersion

import numpy as np


def convert_pt_to_onxx(filepath: str, input_size: tuple):
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE    = 64
    STATE_SPACE   = 9
    ACTION_SPACE  = 9
    EPSILON       = 1.0
    GAMMA         = 0.95
    HIDDEN_SIZE   = 100
    DROPOUT       = 0.15
    TRAIN_START   = 1000
    
    model = DeepQAgentOnnxVersion(
        device        = DEVICE,
        epsilon       = EPSILON, 
        gamma         = GAMMA,
        state_space   = STATE_SPACE, 
        action_space  = ACTION_SPACE, 
        hidden_size   = HIDDEN_SIZE,
        dropout       = DROPOUT,
        train_start   = TRAIN_START,
        batch_size    = BATCH_SIZE
    )
    
    # model = DeepQAgentOnnxVersion() # REPLACE WITH APPRORIATE MODEL

    model.load_state_dict(torch.load(filepath, map_location=torch.device(torch.device("cpu"))))
    model.eval()

    dummy_input = torch.zeros(input_size)
    onnx_model_path = f"trained_models/onnx_models/{filepath.split('/')[-1].split('.')[0]}.onnx"
    torch.onnx.export(
        model, dummy_input, 
        onnx_model_path, 
        verbose=True,
        input_names=["input"], 
        output_names=["output"]
    )
    
    # Check the model
    onnx_model = onnx.load(onnx_model_path)
    
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
    else:
        print("The model is valid!")
        print(onnx.helper.printable_graph(onnx_model.graph))
        
        print("Running Inference Session...")
        ort_session = ort.InferenceSession(onnx_model_path)

        outputs = ort_session.run(
            None,
            {"input": np.array([[0,0,0, 0,0,0, 0,0,0]]).astype(np.float32)},
        )
        print("Inference Results:", outputs[0])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Convert2Onxx',
        description="""
                      Converts a pytorch pt model to an Onxx model that can be 
                      run on a browser or backend server.
                    """,
        epilog=""
    )

    parser.add_argument('filepath')
    parser.add_argument('input_size')
    args = parser.parse_args()

    if type(eval(args.input_size)) is not tuple:
        raise ValueError(
            f"""
              Error: Second argument to must be a valid tensor input size.
              Recieved {type(args.input_size)} instead.
            """
        )

    convert_pt_to_onxx(args.filepath, eval(args.input_size))