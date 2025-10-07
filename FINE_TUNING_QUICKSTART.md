# Quick Start: Fine-Tuned Router

## Train the Model
```bash
# Activate virtual environment
source venv/bin/activate

# Train on brooks_queries.csv using Mac GPU
python train_router.py
```

**Expected Output:**
- Training time: ~3 minutes on Mac GPU
- Test accuracy: 100%
- Model saved to: `./models/fine_tuned_router/`

## Test the Model

### Automated Tests
```bash
python test_fine_tuned_model.py
```

### Interactive Mode
```bash
python test_fine_tuned_model.py --interactive
```

## Use in Code

```python
from test_fine_tuned_model import FineTunedRouter

# Load model
router = FineTunedRouter()

# Predict
result = router.predict("How do I reset my password?")
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Results

- ✅ **100% accuracy** on test set (400 samples)
- ✅ **85% accuracy** on real-world queries
- ✅ Trained on **Mac GPU (MPS)**
- ✅ **No keyword hacks** needed
- ✅ **Fast inference** with confidence scores

## Files Created

1. `train_router.py` - Training script
2. `test_fine_tuned_model.py` - Testing & inference script  
3. `./models/fine_tuned_router/` - Saved model directory
4. `FINE_TUNING_SUMMARY.md` - Detailed results

## What's Next?

You can now:
1. Use the fine-tuned model in your orchestrator
2. Add more training data to improve accuracy
3. Deploy to production
4. Collect feedback and retrain

See `FINE_TUNING_SUMMARY.md` for detailed analysis and improvement suggestions.
