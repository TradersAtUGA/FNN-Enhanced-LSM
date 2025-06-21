# DeepBermuda

DeepBermuda is a neural network-enhanced implementation of the Longstaff-Schwartz Monte Carlo (LSM) method for pricing Bermudan options, with a focus on high-dimensional, over-the-counter (OTC) derivatives. This project extends traditional LSM by replacing polynomial regression with a feedforward neural network to more accurately estimate continuation values in complex multi-asset settings. It is specifically designed to handle basket-style Bermudan options with many underlyings, reflecting real-world OTC structures where early exercise occurs at discrete intervals. The goal is to explore the intersection of deep learning and advanced option pricing for flexible, high-dimensional derivatives.

To see findings and results check `Conclusions.pdf`

## Installation


1. Clone the repository  
   git clone https://github.com/TradersAtUGA/FNN-Enhanced-LSM.git   
   cd your-repo-name

2. Create a virtual environment  
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies  
   pip install -r requirements.txt

4. Run the project  
   python main.py

## Configurable Parameters

The documentation for the params is in the `config_docs.md` file

You can customize the params by editing the `config.yaml` file

## Notes to user

Due to the nature of American options—being exercisable at any time—it is extremely difficult and often unrealistic to accurately price a multi-asset American option basket.

If you attempt to test such a configuration, be aware that it may consume a significant amount of computational resources.

