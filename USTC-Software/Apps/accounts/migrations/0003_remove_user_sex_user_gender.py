# Generated by Django 4.1 on 2024-05-09 14:17

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0002_alter_user_password"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="user",
            name="sex",
        ),
        migrations.AddField(
            model_name="user",
            name="gender",
            field=models.CharField(
                blank=True, max_length=10, null=True, verbose_name="gender"
            ),
        ),
    ]
