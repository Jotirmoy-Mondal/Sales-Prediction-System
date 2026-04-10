import os
import pickle
import numpy as np
from django.shortcuts import render
from django.conf import settings


def load_model():
    """Load the trained ML model from disk."""
    model_path = settings.ML_MODEL_PATH
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None


def home(request):
    """Home page with prediction form."""
    context = {
        'prediction': None,
        'error': None,
        'form_data': {}
    }

    if request.method == 'POST':
        try:
            budget_name = request.POST.get('budget_name', '').strip()
            tv = float(request.POST.get('tv', 0))
            radio = float(request.POST.get('radio', 0))
            newspaper = float(request.POST.get('newspaper', 0))

            # Validate ranges
            if tv < 0 or radio < 0 or newspaper < 0:
                context['error'] = "Budget values cannot be negative."
            elif not budget_name:
                context['error'] = "Please enter a campaign name."
            else:
                model = load_model()
                if model is None:
                    # Fallback: approximate linear regression coefficients
                    # sales ≈ 0.046*TV + 0.188*Radio + 0.001*Newspaper + 2.938
                    prediction = round(0.0458 * tv + 0.1885 * radio + 0.00115 * newspaper + 2.9389, 2)
                else:
                    features = np.array([[tv, radio, newspaper]])
                    prediction = round(float(model.predict(features)[0]), 2)

                context['prediction'] = prediction
                context['form_data'] = {
                    'budget_name': budget_name,
                    'tv': tv,
                    'radio': radio,
                    'newspaper': newspaper,
                }

                # Sales tier
                if prediction < 10:
                    context['tier'] = ('Low', '#e74c3c')
                elif prediction < 16:
                    context['tier'] = ('Moderate', '#f39c12')
                else:
                    context['tier'] = ('High', '#27ae60')

        except ValueError:
            context['error'] = "Please enter valid numeric values for all budget fields."

    return render(request, 'sales_app/index.html', context)
