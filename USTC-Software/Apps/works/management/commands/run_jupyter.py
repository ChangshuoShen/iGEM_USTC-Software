from django.core.management.base import BaseCommand
from jupyter_server import serverapp

class Command(BaseCommand):
    help = 'Run Jupyter server'

    def handle(self, *args, **kwargs):
        serverapp.launch_new_instance(argv=['--notebook-dir=/home/shenc/Desktop/IGEM/iGEM-2024-Coders/USTC-Software/utils'])

