sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model': {
            'values': ['SAGE']
        },
        'learning_rate': {
            'values': [1e-1, 1e-2, 5e-3, 1e-3]
        },
        'dropout': {
            'values': [0.1, 0.2]
        },
        'encoders': {
            'values': [1, 2]
        },
        'conv_layers': {
            'values': [2, 3, 4, 6]
        },
        'hidden_size': {
            'values': [64, 128]
        },
        'decoders': {
            'values': [1, 2]
        },
        'stochastic_weight_avg': {
            'values': [True, False]
        }
    }
}