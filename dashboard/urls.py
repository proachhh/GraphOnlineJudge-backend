from django.urls import path
from .views import DashboardAdminAPI, UserStatsAPI

urlpatterns = [
    path("dashboard/", DashboardAdminAPI.as_view(), name="dashboard_admin_api"),
    path("dashboard/user-stats/", UserStatsAPI.as_view(), name="dashboard_user_stats_api"),
]
