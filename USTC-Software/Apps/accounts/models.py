from django.db import models, connection
from django.contrib.auth.hashers import make_password
from django.utils import timezone
import threading


# 创建一个全局锁对象
user_lock = threading.Lock()

# 用户信息
class User(models.Model):
    MALE = 'male'
    FEMALE = 'female'
    UNCERTAIN = 'uncertain'
    GENDER_CHOICES = [
        (MALE, 'Male'),
        (FEMALE, 'Female'),
        (UNCERTAIN, 'Uncertain'),
    ]
    
    # 一些用户的相关属性
    email = models.EmailField(verbose_name='email', unique=True)
    password = models.CharField(verbose_name='password', max_length=256, blank=False)
    register_date = models.DateTimeField(verbose_name='register_date', default=timezone.now)
    
    avatar = models.ImageField(verbose_name='Avatar', upload_to='avatars/', null=True, blank=True)
    name = models.CharField(verbose_name='name', max_length=64, blank=False, )
    gender = models.CharField(verbose_name='gender', max_length=10, choices=GENDER_CHOICES, default=UNCERTAIN)
    birthday = models.DateField(verbose_name='birthday', null=True, blank=True)
    bio = models.TextField(verbose_name='Bio', null=True, blank=True)
    
    # 下面的一些属性默认都是0,在数据库中建立相关的trigger，从其他table中实现下面内容的更新
    likes_num = models.IntegerField(default=0, verbose_name='likes_num')
    comments_num = models.IntegerField(default=0, verbose_name='comments_num')
    liked_num = models.IntegerField(default=0, verbose_name='liked_num')
    
    is_admin = models.BooleanField(verbose_name='is_admin', default=False)
    is_active = models.BooleanField(verbose_name='is_active', default=True)
    

    def __str__(self):
        return self.email
    
    def edit_profile(self, name=None, gender=None, birthday=None):
        '''
        此处完善用户的信息
        '''
        with user_lock:
            if name:
                self.name = name
            if gender:
                self.gender = gender
            if birthday:
                self.birthday = birthday
            self.save()
    
    @classmethod
    def create_user(cls, name='guest', email=None, raw_password=None, is_admin=False):
        """
        创建用户并设置密码，此时只需要传入名字邮箱、原始密码
        """
        # with user_lock:
        user = cls(
            name=name,
            email=email,
            register_date=timezone.now()
        )

        # 使用 make_password 函数对密码进行哈希处理
        user.set_password(raw_password)
        # 注意之后进行验证的时候使用check_password方法，而不是使用
        
        if is_admin:
            user.set_admin()
        user.save()
        return user

    @classmethod
    def get_all_users(cls):
        # with user_lock:
        return cls.objects.filter(is_active=True).order_by('-register_date')
    
    @classmethod
    def get_user_counts(cls):
        # with user_lock:
        
        total_accounts = cls.objects.count()
        accounts_today = cls.objects.filter(register_date__date=timezone.now().date()).count()
        accounts_yesterday = cls.objects.filter(register_date__date=(timezone.now() - timezone.timedelta(days=1)).date()).count()
        return total_accounts, accounts_today, accounts_yesterday

    @classmethod
    def get_user_by_email(cls, email):
        # with user_lock:    
        try:
            user = cls.objects.get(email=email)
            return user
        except cls.DoesNotExist:
            return None
        
    @classmethod
    def get_user_by_id(cls, user_id):
        # with user_lock:
        try:
            user = cls.objects.get(id=user_id)
            return user
        except cls.DoesNotExist:
            return None


    @classmethod
    def custom_query(cls, *fields, **kwargs):
        '''
        传入位置参数查询相关字段，如果没有传入那就直接查询 * 了
        '''
        with user_lock:
            query = "SELECT {fields} FROM {table_name}".format(
                fields=", ".join(fields) if fields else "*",
                table_name=cls._meta.db_table
                # meta类用于提供元数据metadata，包括数据库的名称，数据库的排序顺序，特定的数据库表选项
            )
            conditions = []     # kwargs传入字段名称-字段值的键值对
            params = []

            for field, value in kwargs.items():
                if value is not None:
                    conditions.append("{field} = %s".format(field=field))
                    params.append(value)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()

            # 一般使用时查询结果至多一个
            if not rows:
                return None
            else:
                result_dict = {}
                for i, field_value in enumerate(rows[0]):
                    field_name = fields[i] if fields else cls._meta.fields[i].name
                    result_dict[field_name] = field_value
                print(result_dict)
                return result_dict

    def set_password(self, raw_password):
        """
        设置用户密码并进行哈希处理
        """
        # with user_lock:
        self.password = make_password(raw_password)
        self.save()
        
    def check_password(self, raw_password):
        """
        检查密码是否匹配
        """
        # with user_lock:
        return raw_password == self.password
    
    def authenticate_user(self, password):
        """
        根据提供的密码验证用户。
        """
        # with user_lock:
        return self.check_password(password)

    def set_admin(self):
        """
        将用户设置为管理员。
        """
        with user_lock:
            self.is_admin = True
            self.save()
        
    def unset_admin(self):
        """
        取消用户的管理员权限。
        """
        with user_lock:
            self.is_admin = False
            self.save()

    
    def is_admin(self):
        """
        检查用户是否是管理员。
        """
        # with user_lock:
        return self.is_admin

    def update_email(self, new_email):
        """
        更新用户的电子邮件地址。
        """
        
        self.email = new_email
        self.save()

    def update_password(self, new_password):
        """
        更新用户的密码。
        """
        self.set_password(new_password)

    def update_avatar(self, new_avatar):
        """
        更新用户的头像。
        """
        with user_lock:
            self.avatar = new_avatar
            self.save()
        
    def deactivate_account(self):
        """
        停用用户帐户。
        """
        with user_lock:
            self.is_active = False
            self.save()

    def delete_account(self):
        """
        删除用户帐户。
        """
        with user_lock:
            self.delete()

    def send_password_reset_email(self):
        """
        发送密码重置电子邮件。
        """
        # 实现逻辑在这里
        pass

    def get_age(self):
        """
        获取用户年龄
        """
        if self.birthdate:
            today = timezone.now()
            age = today.year - self.birthdate.year - (
                        (today.month, today.day) < (self.birthdate.month, self.birthdate.day))
            return age
        else:
            return None

    def get_full_name(self):
        """
        获取用户完整姓名
        """
        # with user_lock:
        return self.name


'''
# 邮箱验证
class EmailVerifyRecord(models.Model):

    # 验证码
    code = models.CharField(max_length=20, verbose_name="verification_code")
    email = models.EmailField(max_length=50, verbose_name="email")

    # 包含注册验证和找回验证
    send_type = models.CharField(verbose_name="send_type", max_length=10,
                                 choices=(("register", "注册"), ("forget", "找回密码")))
    send_time = models.DateTimeField(verbose_name="发送时间", default=timezone.now)

    class Meta:
        verbose_name = u"2. 邮箱验证码"
        verbose_name_plural = verbose_name

    def __unicode__(self):
        return '{0}({1})'.format(self.code, self.email)
'''
