# Generated by Django 5.0.6 on 2024-05-14 12:57

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('redbox_core', '0005_alter_user_password'),
    ]

    operations = [
        migrations.AddField(
            model_name='file',
            name='core_file_uuid',
            field=models.UUIDField(default=uuid.uuid4),
        ),
        migrations.AlterUniqueTogether(
            name='file',
            unique_together={('user', 'core_file_uuid')},
        ),
    ]
