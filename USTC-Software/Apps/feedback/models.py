from django.db import models

class Feedback(models.Model):
    '''
    这里存用户发来的反馈
    '''
    email = models.EmailField()
    satisfaction_level = models.CharField(
        max_length=20,
        choices=[
            ('unhappy', 'Unhappy'),
            ('neutral', 'Neutral'),
            ('satisfied', 'Satisfied'),
        ],
        default='neutral',
    )
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback from {self.name} ({self.email})"

    @classmethod
    def create_feedback(cls, email, satisfaction_level, message):
        feedback = cls(email=email, satisfaction_level = satisfaction_level, message=message)
        feedback.save()
        return feedback

    @classmethod
    def get_all_feedback(cls):
        return cls.objects.all()
