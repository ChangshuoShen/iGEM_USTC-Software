from Apps.forum.models import Comment, Reply
from Apps.accounts.models import User
import random


def create_random_replies():
    
    reply_contents = [
        "I completely agree with you!",
        "Thanks for sharing this information.",
        "Can you provide more details?",
        "That's an interesting point of view.",
        "I had the same question.",
        "I don't think that's correct.",
        "Great insight, thanks!",
        "I have a different opinion on this."
    ]
    
    comments = Comment.objects.all()
    users = User.objects.all()

    for comment in comments:
        if not users:
            print("No users found in the database.")
            return
        
        user = random.choice(users)
        reply_content = random.choice(reply_contents)
        Reply.create_reply(comment=comment, user=user, content=reply_content)

    print(f"Created random replies for {comments.count()} comments.")
    print("12334444")

