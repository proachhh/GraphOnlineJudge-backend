# -*- coding: utf-8 -*-
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models

import utils.models
import utils.shortcuts


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Feedback',
            fields=[
                ('id', models.CharField(db_index=True, default=utils.shortcuts.rand_str, max_length=32, primary_key=True, serialize=False)),
                ('username', models.CharField(max_length=32)),
                ('title', models.CharField(max_length=128)),
                ('content', utils.models.RichTextField(blank=True, null=True)),
                ('resolved', models.BooleanField(default=False)),
                ('admin_note', models.TextField(blank=True, null=True)),
                ('create_time', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'feedback',
                'ordering': ('-create_time',),
            },
        ),
    ]
