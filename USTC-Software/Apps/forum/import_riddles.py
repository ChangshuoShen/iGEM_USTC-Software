import csv
from django.core.management.base import BaseCommand
from forum.models import riddles  # 替换成你的模型类

class Command(BaseCommand):
    help = 'Import riddles from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file containing riddles')

    def handle(self, *args, **options):
        csv_file = options['csv_file']
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                main_category = row['Main_category']
                riddle_text = row['Riddle']
                answer_text = row['Answer']
                difficulty = row['Difficulty']
                
                # 创建谜语实例并保存到数据库中
                riddles.objects.create(
                    main_category=main_category,
                    riddle_text=riddle_text,
                    answer_text=answer_text,
                    difficulty=difficulty
                )
        self.stdout.write(self.style.SUCCESS('Riddles imported successfully'))
