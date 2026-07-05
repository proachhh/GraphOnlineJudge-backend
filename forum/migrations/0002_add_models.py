from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('forum', '0001_initial'),
    ]
    operations = [
        migrations.AddField(
            model_name='comment',
            name='like_count',
            field=models.IntegerField(default=0),
        ),
        migrations.CreateModel(
            name='CommentLike',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('create_time', models.DateTimeField(auto_now_add=True)),
                ('comment', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='likes', to='forum.comment')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={'db_table': 'forum_comment_like', 'unique_together': {('user', 'comment')}},
        ),
        migrations.CreateModel(
            name='Report',
            fields=[
                ('id', models.CharField(db_index=True, default='', max_length=32, primary_key=True, serialize=False)),
                ('target_type', models.CharField(max_length=16)),
                ('target_id', models.CharField(max_length=32)),
                ('reason', models.CharField(default='other', max_length=32)),
                ('detail', models.TextField(blank=True, null=True)),
                ('status', models.CharField(default='pending', max_length=16)),
                ('handle_note', models.TextField(blank=True, null=True)),
                ('create_time', models.DateTimeField(auto_now_add=True)),
                ('handle_time', models.DateTimeField(null=True, blank=True)),
                ('handled_by', models.ForeignKey(null=True, blank=True, on_delete=django.db.models.deletion.SET_NULL, related_name='reports_handled', to=settings.AUTH_USER_MODEL)),
                ('reporter', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='reports_made', to=settings.AUTH_USER_MODEL)),
            ],
            options={'db_table': 'forum_report', 'ordering': ('-create_time',)},
        ),
        migrations.CreateModel(
            name='UserMute',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('reason', models.TextField()),
                ('muted_until', models.DateTimeField(null=True, blank=True)),
                ('create_time', models.DateTimeField(auto_now_add=True)),
                ('muted_by', models.ForeignKey(null=True, blank=True, on_delete=django.db.models.deletion.SET_NULL, related_name='mutes_given', to=settings.AUTH_USER_MODEL)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='mute_record', to=settings.AUTH_USER_MODEL)),
            ],
            options={'db_table': 'forum_user_mute'},
        ),
    ]
