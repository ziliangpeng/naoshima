from Crypto.Hash import MD5
import msg

def hash(message):
    h = MD5.new()
    h.update(message)
    return h.hexdigest()

def run(silence=True):
    message = msg.MESSAGE
    digest = hash(message)

    if not silence:
        print(message)
        print(digest)

if __name__ == '__main__':
    run(silence=False)
