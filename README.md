# Self-Driving Car in Grand Theft Auto: San Andreas

This project implements a self-driving car system in GTA: San Andreas using computer vision and deep learning. The AI model learns to drive by capturing game footage and predicting appropriate controls, enhanced with lane detection for better navigation.

## Features

- Real-time screen capture and processing
- Deep learning model based on ResNet architecture
- Automated keyboard control simulation
- Performance monitoring and visualization
- Debug mode for development

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- PyGetWindow
- PyDirectInput
- Keyboard
- NumPy
- GTA: San Andreas game installed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AshyrMalik/Self-Driving-Car-in-Grand-Theft-Auto-San-Andreas.git
cd Self-Driving-Car-in-Grand-Theft-Auto-San-Andreas
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── model_loader.py      # Model handling and predictions
├── model_trainer.py     # Training pipeline for the AI model
├── screen_capture.py    # Game screen capture functionality
├── game_control.py      # Game control interface
├── config.py           # Configuration settings
└── main.py            # Main execution script
```

## Usage

1. Launch GTA: San Andreas
2. Run the self-driving system:
```bash
python main.py
```

### Controls
- `Q`: Quit the program
- `P`: Pause/Resume the AI
- `V`: Toggle visualization
- `D`: Toggle debug information

## How It Works

1. **Screen Capture**: The system captures real-time footage from the game window using PyGetWindow.

2. **Image Processing**:
   - Gets resolution for the game window
   - Captures frames are processed and normalized
   - Processed images are fed into the neural network

4. **Model Prediction**:
   - ResNet-based model predicts appropriate driving actions
   - Thresholds the models prediction for each class
   - Outputs control signals (W, A, S, D)

5. **Game Control**:
   - Predictions are converted to keyboard inputs using PyDirectInput
   - Controls are applied in real-time to navigate the vehicle

## Model Training

The model was trained on a dataset of gameplay footage with corresponding control inputs. ResNet architecture was chosen for its superior performance in this application.

### Training Process:
1. Data collection from manual gameplay
2. Image preprocessing and augmentation
3. Model training with labeled data
4. Performance evaluation and tuning

## Performance

- Real-time processing at stable FPS
- Reliable lane detection and following
- Smooth control transitions
- Automated driving in various conditions

## Limitations

- Performance depends on game resolution and frame rate
- May require adjustments for different vehicle types
- Best performance on clear roads with visible lanes
- Requires consistent lighting conditions

## Future Improvements

- [ ] Implement object detection for traffic and obstacles
- [ ] Improve night driving capabilities
- [ ] Add collision prediction and avoidance
- [ ] Lane detection system for improved navigation
      
## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to Kaggle for the GTA:SA driving dataset
- ResNet architecture implementation
- OpenCV community for computer vision tools

## Author

Ashyr Malik

## Contact

- GitHub: [@AshyrMalik](https://github.com/AshyrMalik)
