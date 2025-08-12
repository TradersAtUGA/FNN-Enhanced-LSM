echo "=== Starting Test 1 ==="
python main.py --config config.yaml
echo "=== Finished Test 1 ==="

echo "=== Starting Test 2 ==="
python main.py --config config1d.yaml
echo "=== Finished Test 2 ==="

echo "=== Starting Test 3 ==="
python main.py --config configEuro.yaml
echo "=== Finished Test 3 ==="