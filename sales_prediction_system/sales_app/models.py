from django.db import models


class PredictionHistory(models.Model):
    """Store prediction history."""
    tv_budget = models.FloatField()
    radio_budget = models.FloatField()
    newspaper_budget = models.FloatField()
    predicted_sales = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Prediction Histories'

    def __str__(self):
        return f"TV={self.tv_budget}, Radio={self.radio_budget} → Sales={self.predicted_sales}"
