# Generated by Django 4.2.11 on 2024-04-17 15:04

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('redbox_core', '0002_user_invite_accepted_at_user_invited_at_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChatHistory',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
                ('name', models.TextField(max_length=1024)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='File',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
                ('processing_status', models.CharField(choices=[('uploaded', 'Uploaded'), ('parsing', 'Parsing'), ('chunking', 'Chunking'), ('embedding', 'Embedding'), ('indexing', 'Indexing'), ('complete', 'Complete')])),
                ('original_file', models.FileField(storage='redbox-storage-dev', upload_to='')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='ChatMessage',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('modified_at', models.DateTimeField(auto_now=True)),
                ('text', models.TextField(max_length=32768)),
                ('role', models.TextField(max_length=1024)),
                ('chat_history', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='redbox_core.chathistory')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.AddField(
            model_name='chathistory',
            name='files_received',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='files_received', to='redbox_core.file'),
        ),
        migrations.AddField(
            model_name='chathistory',
            name='files_retrieved',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='files_retrieved', to='redbox_core.file'),
        ),
        migrations.AddField(
            model_name='chathistory',
            name='users',
            field=models.ManyToManyField(to=settings.AUTH_USER_MODEL),
        ),
    ]
