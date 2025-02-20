# class EventBus:
#     _subscribers = {}
#
#     @classmethod
#     def subscribe(cls, event_name, callback):
#         if event_name not in cls._subscribers:
#             cls._subscribers[event_name] = []
#         cls._subscribers[event_name].append(callback)
#
#     @classmethod
#     def publish(cls, event_name, data):
#         if event_name in cls._subscribers:
#             for callback in cls._subscribers[event_name]:
#                 callback(data)