"""
Browser Notification Utilities
"""
import streamlit.components.v1 as components


def request_notification_permission():
    """Request browser notification permission on page load"""
    html = """
    <script>
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
    </script>
    """
    components.html(html, height=0)


def send_browser_notification(title: str, body: str):
    """
    Send browser notification

    Args:
        title: Notification title
        body: Notification body text
    """
    # Escape single quotes in title and body
    title = title.replace("'", "\\'")
    body = body.replace("'", "\\'")

    html = f"""
    <script>
    if ('Notification' in window && Notification.permission === 'granted') {{
        new Notification('{title}', {{
            body: '{body}',
            icon: '⚠️',
            requireInteraction: false
        }});
    }}
    </script>
    """
    components.html(html, height=0)
