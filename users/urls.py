from django.urls import path
from .views import home, profile, RegisterView, CustomLoginView
from .forms import LoginForm


urlpatterns = [
    path('', CustomLoginView.as_view(redirect_authenticated_user=True, template_name='users/login.html',
                                           authentication_form=LoginForm), name='login'),
    path('register/', RegisterView.as_view(), name='users-register'),
    path('profile/', profile, name='users-profile'),
]
