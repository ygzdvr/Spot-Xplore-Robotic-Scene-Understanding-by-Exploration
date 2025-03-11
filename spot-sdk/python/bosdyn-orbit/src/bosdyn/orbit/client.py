# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

""" The client uses a web API to send HTTPs requests to a number of REStful endpoints using the Requests library.
"""
from typing import Dict, Iterable

import requests

import bosdyn.orbit.utils as utils
from bosdyn.orbit.exceptions import UnauthenticatedClientError

DEFAULT_HEADERS = {'Accept': 'application/json'}
OCTET_HEADER = {'Content-type': 'application/octet-stream', 'Accept': 'application/octet-stream'}


class Client():
    """Client for the Orbit web API"""

    def __init__(self, hostname: str, verify: bool = True, cert: str = None):
        """ Initializes the attributes of the Client object.

            Args:
                hostname: the IP address associated with the instance
                verify(path to a CA bundle or Boolean): controls whether we verify the server’s TLS certificate
                    Note that verify=False makes your application vulnerable to man-in-the-middle (MitM) attacks.
                    Defaults to True.
                cert(.pem file or a tuple with ('cert', 'key') pair): a local cert to use as client side certificate
                    Note that the private key to your local certificate must be unencrypted because Requests does not support using encrypted keys.
                    Defaults to None. For additional configurations, use the member Session object "_session" in accordance with Requests library.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
        """
        # The hostname of the instance
        self._hostname = hostname
        # Set a Session object to persist certain parameters across requests
        self._session = requests.Session()
        # Set SSL verification strategy
        self._session.verify = verify
        # Client Side Certificates
        self._session.cert = cert
        # Initialize session
        self._session.get(f'https://{self._hostname}')
        # Set default headers for self._session
        self._session.headers.update(DEFAULT_HEADERS)
        # Set x-csrf-token for self._session
        self._session.headers.update({'x-csrf-token': self._session.cookies['x-csrf-token']})
        # Flag indicating that the Client is authenticated
        self._is_authenticated = False

    def authenticate_with_api_token(self, api_token: str = None) -> requests.Response:
        """ Authorizes the client using the provided API token obtained from the instance.
            Must call before using other client functions.

            Args:
                api_token: the API token obtained from the instance.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
        """
        if not api_token:
            api_token = utils.get_api_token()
        # Set API token for self._session
        self._session.headers.update({'Authorization': 'Bearer ' + api_token})
        # Check the validity of the API token
        authenticate_response = self._session.get(
            f'https://{self._hostname}/api/v0/api_token/authenticate')
        if authenticate_response.ok:
            self._is_authenticated = True
        else:
            print('Client: Login failed: {} Please, obtain a valid API token from the instance!'.
                  format(authenticate_response.text))
            # Remove the invalid API token from session headers
            self._session.headers.pop('Authorization')
        return authenticate_response

    def get_resource(self, path: str, **kwargs) -> requests.Response:
        """ Base function for getting a resource in /api/v0/.

            Args:
                path: the path appended to /api/v0/.
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        if not self._is_authenticated:
            raise UnauthenticatedClientError()
        return self._session.get(f'https://{self._hostname}/api/v0/{path}/', **kwargs)

    def post_resource(self, path: str, **kwargs) -> requests.Response:
        """ Base function for posting a resource in /api/v0/.

            Args:
                path: the path appended to /api/v0/
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                The response associated with the post request.
        """
        if not self._is_authenticated and path != "login":
            raise UnauthenticatedClientError()
        return self._session.post(f'https://{self._hostname}/api/v0/{path}', **kwargs)

    def patch_resource(self, path: str, **kwargs) -> requests.Response:
        """ Base function for patching a resource in /api/v0/

            Args:
                path: the path appended to /api/v0/
                kwargs(**): a variable number of keyword arguments for the patch request
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                The response associated with the patch request.
        """
        if not self._is_authenticated and path != "login":
            raise UnauthenticatedClientError()
        return self._session.patch(f'https://{self._hostname}/api/v0/{path}', **kwargs)

    def delete_resource(self, path: str, **kwargs) -> requests.Response:
        """ Base function for deleting a resource in /api/v0/.

            Args:
                path: the path appended to /api/v0/
                kwargs(**): a variable number of keyword arguments for the delete request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                The response associated with the delete request
        """
        if not self._is_authenticated:
            raise UnauthenticatedClientError()
        return self._session.delete(f'https://{self._hostname}/api/v0/{path}', **kwargs)

    def get_version(self, **kwargs) -> requests.Response:
        """ Retrieves version info.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("version", **kwargs)

    def get_system_time(self, **kwargs) -> requests.Response:
        """ Returns the current system time.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("settings/system-time", **kwargs)

    def get_robots(self, **kwargs) -> requests.Response:
        """ Returns robots on the specified instance.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("robots", **kwargs)

    def get_robot_by_hostname(self, hostname: str, **kwargs) -> requests.Response:
        """ Returns a robot on given a hostname of a specific robot.

            Args:
                hostname: the IP address associated with the desired robot on the instance.
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource(f'robots/{hostname}', **kwargs)

    def get_site_walks(self, **kwargs) -> requests.Response:
        """ Returns site walks on the specified instance.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("site_walks", **kwargs)

    def get_site_walk_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Given a SiteWalk uuid, returns a SiteWalk on the specified instance.

            Args:
                uuid: the ID associated with the SiteWalk.
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource(f'site_walks/{uuid}', **kwargs)

    def get_site_walk_archive_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Returns SiteWalk as a zip archive which represents a collection of graph and mission data

            Args:
                kwargs(**): a variable number of keyword arguments for the get request
            Raises:
                RequestExceptions: exceptions thrown by the Requests library
                UnauthenticatedClientError:  indicates that the client is not authenticated properly
            Returns:
                requests.Response: the response associated with the get request
        """
        return self.get_resource(f"site_walks/archive", params={"uuids": uuid}, **kwargs)

    def get_site_elements(self, **kwargs) -> requests.Response:
        """ Returns site elements on the specified instance.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("site_elements", **kwargs)

    def get_site_element_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Given a SiteElement uuid, returns a SiteElement on the specified instance.

            Args:
                uuid: the ID associated with the SiteElement.
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource(f'site_elements/{uuid}', **kwargs)

    def get_site_docks(self, **kwargs) -> requests.Response:
        """ Returns site docks on the specified instance.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("site_docks", **kwargs)

    def get_site_dock_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Given a SiteDock uuid, returns a SiteDock on the specified instance.

            Args:
                uuid: the ID associated with the SiteDock
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource(f'site_docks/{uuid}', **kwargs)

    def get_calendar(self, **kwargs) -> requests.Response:
        """ Returns calendar events on the specified instance.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("calendar/schedule", **kwargs)

    def get_run_events(self, **kwargs) -> requests.Response:
        """ Given a dictionary of query params, returns run events.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("run_events", **kwargs)

    def get_run_event_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Given a runEventUuid, returns a run event.

            Args:
                uuid: the ID associated with the run event.
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource(f'run_events/{uuid}', **kwargs)

    def get_run_captures(self, **kwargs) -> requests.Response:
        """ Given a dictionary of query params, returns run captures.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("run_captures", **kwargs)

    def get_run_capture_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Given a runCaptureUuid, returns a run capture.

            Args:
                uuid: the ID associated with the run capture
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource(f'run_captures/{uuid}', **kwargs)

    def get_runs(self, **kwargs) -> requests.Response:
        """ Given a dictionary of query params, returns runs.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("runs", **kwargs)

    def get_run_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Given a runUuid, returns a run.

            Args:
                uuid: the ID associated with the run
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource(f'runs/{uuid}', **kwargs)

    def get_run_archives_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Given a runUuid, returns run archives.

            Args:
                uuid: the ID associated with the run
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource(f'run_archives/{uuid}', **kwargs)

    def get_image(self, url: str, **kwargs) -> 'urllib3.response.HTTPResponse':
        """ Given a data capture url, returns a decoded image.

            Args:
                url: the url associated with the data capture in the form of https://hostname + RunCapture["dataUrl"].
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                urllib3.response.HTTPResponse: the decoded response associated with the get request
        """
        if not self._is_authenticated:
            raise UnauthenticatedClientError()
        response = self._session.get(url, stream=True, **kwargs)
        response.raise_for_status()
        response.raw.decode_content = True
        return response.raw

    def get_image_response(self, url: str, **kwargs) -> requests.Response:
        """ Given a data capture url, returns an image response.

            Args:
                url: the url associated with the data capture in the form of https://hostname + RunCapture["dataUrl"]
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the image response associated with the get request
        """
        if not self._is_authenticated:
            raise UnauthenticatedClientError()
        response = self._session.get(url, stream=True, **kwargs)
        return response

    def get_webhook(self, **kwargs) -> requests.Response:
        """ Returns webhook on the specified instance.

            Args:
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource("webhooks", **kwargs)

    def get_webhook_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Given a uuid, returns a specific webhook instance.

            Args:
                uuid: the ID associated with the webhook
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.get_resource(f'webhooks/{uuid}', **kwargs)

    def get_robot_info(self, robot_nickname: str, **kwargs) -> requests.Response:
        """ Given a robot nickname, returns information about the robot.

            Args:
                robot_nickname: the nickname of the robot
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.get_resource(f'robot-session/{robot_nickname}/session', **kwargs)

    def post_export_as_walk(self, site_walk_uuid: str, **kwargs) -> requests.Response:
        """ Given a SiteWalk uuid, it exports the walks_pb2.Walk equivalent.

            Args:
                site_walk_uuid: the ID associated with the SiteWalk.
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError: indicates that the client is not authenticated properly
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource(f'site_walks/export_as_walk',
                                  json={"siteWalkUuid": site_walk_uuid}, **kwargs)

    def post_import_from_walk(self, **kwargs) -> requests.Response:
        """ Given a walk data, imports it to the specified instance.

            Args:
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("site_walks/import_from_walk", **kwargs)

    def post_site_element(self, **kwargs) -> requests.Response:
        """ Create a SiteElement. It also updates a pre-existing SiteElement using the associated UUID.

            Args:
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("site_elements", **kwargs)

    def post_site_walk(self, **kwargs) -> requests.Response:
        """ Create a SiteWalk. It also updates a pre-existing SiteWalk using the associated UUID.

            Args:
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("site_walks", **kwargs)

    def post_site_dock(self, **kwargs) -> requests.Response:
        """ Create a SiteElement. It also updates a pre-existing SiteDock using the associated UUID.

            Args:
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("site_docks", **kwargs)

    def post_robot(self, **kwargs) -> requests.Response:
        """ Add a robot to the specified instance.

            Args:
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("robots", **kwargs)

    def post_calendar_event(self, nickname: str = None, time_ms: int = None, repeat_ms: int = None,
                            mission_id: str = None, force_acquire_estop: bool = None,
                            require_docked: bool = None, schedule_name: str = None,
                            blackout_times: Iterable[Dict[str,
                                                          int]] = None, disable_reason: str = None,
                            event_id: str = None, **kwargs) -> requests.Response:
        """  This function serves two purposes. It creates a new calendar event on using the following arguments
             when Event ID is not specified. When the Event ID associated with a pre-existing calendar event is specified,
             the function overwrites the attributes of the pre-existing calendar event.

            Args:
                nickname: the name associated with the robot.
                time_ms: the first kickoff time in terms of milliseconds since epoch.
                repeat_ms:the delay time in milliseconds for repeating calendar events.
                mission_id: the UUID associated with the mission( also known as SiteWalk).
                force_acquire_estop: instructs the system to force acquire the estop when the mission kicks off.
                require_docked: determines whether the event will require the robot to be docked to start.
                schedule_name: the desired name of the calendar event.
                blackout_times: a specification for a time period over the course of a week when a schedule should not run
                                  specified as list of a dictionary defined as {"startMs": <int>, "endMs" : <int>}
                                  with startMs (inclusive) being the millisecond offset from the beginning of the week (Sunday) when this blackout period starts
                                  and endMs (exclusive) being the millisecond offset from beginning of the week(Sunday) when this blackout period ends.
                disable_reason: (optional) a reason for disabling the calendar event.
                event_id: the auto-generated ID for a calendar event that is already posted on the instance.
                            This is only useful when editing a pre-existing calendar event.
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        # Check if the input contains the json param that is constructed outside the function
        if 'json' in kwargs:
            return self.post_resource("calendar/schedule", **kwargs)
        # Construct payload based on provided inputs
        payload = {
            "agent": {
                "nickname": nickname
            },
            "schedule": {
                "timeMs": time_ms,
                "repeatMs": repeat_ms,
                "blackouts": blackout_times,
                "disableReason": disable_reason,
            },
            "task": {
                "missionId": mission_id,
                "forceAcquireEstop": force_acquire_estop,
                "requireDocked": require_docked,
            },
            "eventMetadata": {
                "name": schedule_name,
                "eventId": event_id
            },
        }
        return self.post_resource("calendar/schedule", json=payload, **kwargs)

    def post_calendar_events_disable_all(self, disable_reason: str, **kwargs) -> requests.Response:
        """ Disable all scheduled missions.

            Args:
                disable_reason: Reason for disabling all scheduled missions.
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("calendar/disable-enable", json={"disableReason": disable_reason},
                                  **kwargs)

    def post_calendar_event_disable_by_id(self, event_id: str, disable_reason: str,
                                          **kwargs) -> requests.Response:
        """ Disable specific scheduled mission by event ID.

            Args:
                event_id: eventId associated with a mission to disable.
                disable_reason: Reason for disabling a scheduled mission.
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("calendar/disable-enable", json={
            "disableReason": disable_reason,
            "eventId": event_id
        }, **kwargs)

    def post_calendar_events_enable_all(self, **kwargs) -> requests.Response:
        """ Enable all scheduled missions.

            Args:
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("calendar/disable-enable", json={"disableReason": ""}, **kwargs)

    def post_calendar_event_enable_by_id(self, event_id: str, **kwargs) -> requests.Response:
        """ Enable specific scheduled mission by event ID.

            Args:
                event_id: eventId associated with a mission to enable.
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("calendar/disable-enable", json={
            "disableReason": "",
            "eventId": event_id
        }, **kwargs)

    def post_webhook(self, **kwargs) -> requests.Response:
        """ Create a webhook instance.

            Args:
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource("webhooks", **kwargs)

    def post_webhook_by_id(self, uuid: str, **kwargs) -> requests.Response:
        """ Update an existing webhook instance.

            Args:
                uuid: the ID associated with the desired webhook instance.
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource(f'webhooks/{uuid}', **kwargs)

    def post_return_to_dock_mission(self, robot_nickname: str, site_dock_uuid: str,
                                    **kwargs) -> requests.Response:
        """ Generate a mission to send the robot back to the dock.

            Args:
                robot_nickname: the nickname of the robot.
                site_dock_uuid: the uuid of the dock to send robot to.
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        return self.post_resource('graph/send-robot', json={
            "nickname": robot_nickname,
            "siteDockUuid": site_dock_uuid
        }, **kwargs)

    def post_dispatch_mission_to_robot(self, robot_nickname: str, driver_id: str, mission_uuid: str,
                                       delete_mission: bool, force_acquire_estop: bool,
                                       skip_initialization: bool, **kwargs) -> requests.Response:
        """ Dispatch the robot to a mission given a mission uuid.

            Args:
                robot_nickname: the nickname of the robot.
                driver_id: the current driver ID of the mission.
                mission_uuid: uuid of the mission(also known as SiteWalk) to dispatch.
                delete_mission: whether to delete the mission after playback.
                force_acquire_estop: whether to force acquire E-stop from the previous client.
                skip_initialization: whether to skip initialization when starting the return to dock mission.
                kwargs(**): a variable number of keyword arguments for the post request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the post request.
        """
        # Payload required for dispatching a mission
        payload = {
            "agent": {
                "nickname": robot_nickname
            },
            "schedule": {
                "timeMs": {
                    "low": 1,
                    "high": 0,
                    "unsigned": False
                },
                "repeatMs": {
                    "low": 0,
                    "high": 0,
                    "unsigned": False
                }
            },
            "task": {
                "missionId": mission_uuid,
                "forceAcquireEstop": force_acquire_estop,
                "deleteMission": delete_mission,
                "requireDocked": False,
                "skipInitialization": skip_initialization
            },
            "eventMetadata": {
                "name": "API Triggered Mission"
            }
        }
        return self.post_resource(
            f'calendar/mission/dispatch/{robot_nickname}?currentDriverId={driver_id}', json=payload,
            **kwargs)

    def post_backup_task(self, include_missions: bool, include_captures: bool,
                         **kwargs) -> requests.Response:
        """ Starts creating a backup zip file.
        
            Args: 
                include_missions: Specifies whether to include missions and maps in the backup.
                include_captures: Specifies whether to include all inspection data captures in the backup.
                **kwargs: Additional keyword arguments for the backup request.
            Raises:
                RequestExceptions: Exceptions thrown by the Requests library.
                UnauthenticatedClientError: Indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the backup request.
        """
        payload = {
            "includeMissions": include_missions,
            "includeCaptures": include_captures,
        }
        return self.post_resource(f'backup_tasks/', json=payload, **kwargs)

    def patch_bulk_close_anomalies(self, element_ids: list[str], **kwargs) -> requests.Response:
        """ Bulk close Anomalies by Element ID.

            Args:
                element_ids: the element ids of each anomaly to be closed.
                kwargs(**): a variable number of keyword arguments for the patch request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the patch request.
        """
        return self.patch_resource('anomalies', json={
            "command": "close",
            "elementIds": element_ids
        }, **kwargs)

    def patch_anomaly_by_id(self, anomaly_uuid: str, patched_fields: dict,
                            **kwargs) -> requests.Response:
        """ Patch an Anomaly by uuid.

            Args:
                anomaly_uuid: The uuid of the anomaly to patch fields in.
                patched_fields: A dictionary of fields and new values to change in the specified anomaly.
                kwargs(**): a variable number of keyword arguments for the patch request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: The response associated with the patch request.
        """
        return self.patch_resource(f'anomalies/{anomaly_uuid}', json=patched_fields, **kwargs)

    def delete_site_walk(self, uuid: str, **kwargs) -> requests.Response:
        """ Given a SiteWalk uuid, deletes the SiteWalk associated with the uuid on the specified instance.

            Args:
                uuid: the ID associated with the desired SiteWalk
                kwargs(**): a variable number of keyword arguments for the delete request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the delete request.
        """
        return self.delete_resource(f'site_walks/{uuid}', **kwargs)

    def delete_robot(self, robot_hostname: str, **kwargs) -> requests.Response:
        """ Given a robot hostname, deletes the robot associated with the hostname on the specified instance

            Args:
                robot_hostname: the IP address associated with the robot.
                kwargs(**): a variable number of keyword arguments for the delete request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the delete request.
        """
        return self.delete_resource(f'robots/{robot_hostname}', **kwargs)

    def delete_calendar_event(self, event_id: str, **kwargs) -> requests.Response:
        """ Delete the specified calendar event on the specified instance.

            Args:
                event_id(string): the ID associated with the calendar event.
                kwargs(**): a variable number of keyword arguments for the delete request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the delete request.
        """
        return self.delete_resource(f'calendar/schedule/{event_id}', **kwargs)

    def delete_webhook(self, uuid: str, **kwargs) -> requests.Response:
        """ Delete the specified webhook instance on the specified instance.

            Args:
                uuid: the ID associated with the desired webhook.
                kwargs(**): a variable number of keyword arguments for the delete request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the delete request.
        """
        return self.delete_resource(f'webhooks/{uuid}', **kwargs)

    def delete_backup(self, task_id: str, **kwargs):
        """ Deletes the backup zip file from the Orbit instance. 

            Args:
                hostname: the IP address associated with the desired robot on the instance.
                kwargs(**): a variable number of keyword arguments for the get request.
            Raises:
                RequestExceptions: exceptions thrown by the Requests library.
                UnauthenticatedClientError:  indicates that the client is not authenticated properly.
            Returns:
                requests.Response: the response associated with the get request.
        """
        return self.delete_resource(f'backups/{task_id}', **kwargs)


def create_client(options: 'argparse.Namespace') -> 'bosdyn.orbit.client.Client':
    """ Creates a client object.

        Args:
            options: User input containing hostname, verification, and certification info.
        Returns:
            client: 'bosdyn.orbit.client.Client' object
    """
    # Determine the value for the argument "verify"
    if options.verify in ["True", "False"]:
        verify = options.verify == "True"
    else:
        print(
            "The provided value for the argument verify [{}] is not either 'True' or 'False'. Assuming verify is set to 'path/to/CA bundle'"
            .format(options.verify))
        verify = options.verify

    # A client object represents a single instance.
    client = Client(hostname=options.hostname, verify=verify, cert=options.cert)

    # The client needs to be authenticated before using its functions
    client.authenticate_with_api_token()

    return client
