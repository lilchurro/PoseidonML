'''
Trains and saves an instance of the one layer feedforward model on the
data directory specified by the '-P' argument ('/pcaps' by default). The
model is saved to a location specified by the -w parameter
('models/OneLayerModel' by default).
'''
import argparse

from poseidonml.config import get_config
from poseidonml.Model import Model
from sklearn.neural_network import MLPClassifier



default_pcap_dir = '/pcaps'
default_model_file = 'models/OneLayerModel.pkl'
default_conf_file = 'opts/config.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default=default_conf_file,
                        help='model\'s config file')
    parser.add_argument('--pcaps', '-P', default=default_pcap_dir,
                        help='pcap directory to train on (e.g., /pcaps)')
    parser.add_argument('--save', '-w', default=default_model_file,
                        help='path to save model (e.g., models/OneLayerModel.json)')

    args = parser.parse_args()

    # Load model params from config
    config = get_config(args.config)
    duration = config['duration']
    hidden_size = config['state size']
    labels = config['labels']

    # Get the data directory
    data_dir = args.pcaps

    m = MLPClassifier(
        (hidden_size),
        alpha=0.1,
        activation='relu',
        max_iter=1000
    )

    # Initialize the model
    model = Model(
        duration=duration,
        hidden_size=hidden_size,
        labels=labels,
        model=m,
        model_type='OneLayer'
    )
    # Train the model
    model.train(data_dir)
    # Save the model to the specified path
    model.save(args.save, jsn=True)

if __name__ == '__main__':
    main()
