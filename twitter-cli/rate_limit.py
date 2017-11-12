import auth


api = auth.auth()

print(api.rate_limit_status())
