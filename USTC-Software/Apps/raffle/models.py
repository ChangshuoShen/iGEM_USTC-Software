from django.db import models
import random

class Prize(models.Model):
    name = models.CharField(max_length=100, verbose_name='name')
    quantity = models.PositiveIntegerField(verbose_name='quantity')
    
    def __str__(self):
        return f"{self.name} - {self.quantity}"
    @classmethod
    def update_prize(cls, prize_id, new_quantity):
        try:
            prize = cls.objects.get(pk=prize_id)
            prize.quantity = new_quantity
            prize.save()
            return True
        except cls.DoesNotExist:
            return False

    @classmethod
    def draw_prize(cls):
        prizes = cls.objects.all()
        total_quantity = sum(prize.quantity for prize in prizes)
        if total_quantity == 0:
            return None
        random_number = random.uniform(0, total_quantity)
        cumulative_quantity = 0
        for prize in prizes:
            cumulative_quantity += prize.quantity
            if random_number <= cumulative_quantity:
                return prize
        return None

    @classmethod
    def get_all_prizes(cls):
        return cls.objects.all()
    
    @classmethod
    def clear_all_prizes(cls):
        # 一键清除所有奖品
        cls.objects.all().delete()

    @classmethod
    def create_from_dict(cls, prizes_dict):
        # 从字典中创建全新的奖品数据库
        cls.objects.all().delete()  # 清除现有奖品
        for name, quantity in prizes_dict.items():
            cls.objects.create(name=name, quantity=quantity)