import torch
import argparse
from models.ReinforcementLearning.DeepQAgent import DeepQAgent


def convert_pt_to_onxx(filepath: str, input_size: tuple):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # BATCH_SIZE    = 64
    # STATE_SPACE   = 9
    # ACTION_SPACE  = 9
    # EPSILON       = 1.0
    # GAMMA         = 0.95
    # HIDDEN_SIZE   = 100
    # DROPOUT       = 0.15
    # TRAIN_START   = 1000

    # model = DeepQAgent(
    #     device        = DEVICE,
    #     epsilon       = EPSILON, 
    #     gamma         = GAMMA,
    #     state_space   = STATE_SPACE, 
    #     action_space  = ACTION_SPACE, 
    #     hidden_size   = HIDDEN_SIZE,
    #     dropout       = DROPOUT,
    #     train_start   = TRAIN_START,
    #     batch_size    = BATCH_SIZE
    # )
    
    model = DeepQAgent() # REPLACE WITH APPRORIATE MODEL

    model.load_state_dict(torch.load(filepath, map_location=torch.device(DEVICE)))
    model.eval()

    dummy_input = torch.zeros(input_size)
    torch.onnx.export(
        model, dummy_input, 
        f"{filepath.split('.')[0]}.onnx", 
        verbose=True
    )

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