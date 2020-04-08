from django.contrib import admin
from .models import Database

# Register your models here.


class DatabaseAdmin(admin.ModelAdmin):
    list_display = ('pk', 'question', 'answer')
    list_filter = ('question', 'answer')
    search_fields = ['pk', 'question', 'answer']

    def account_query(self, obj):
        return obj.account

    account_query.admin_order_field = 'account_query'

    pass


admin.site.register(Database, DatabaseAdmin)
