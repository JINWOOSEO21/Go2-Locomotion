"""데모용 키보드 텔레오퍼레이션 입력기.

이 파일이 존재하는 이유:
  demo.py 는 원래 게임패드(carb.input.GamepadInput)만 지원했다. carb 의 입력은
  Kit 앱 윈도우(omni.appwindow)에 붙어 있어서, 헤드리스로 띄우면 앱 윈도우가 없고
  따라서 키보드/게임패드 이벤트가 아예 발생하지 않는다. GUI 로 띄우더라도 그 키보드는
  이 머신에 물려 있는 물리 키보드지 SSH 세션의 키보드가 아니다.

그래서 입력 경로를 두 개 둔다.

  1) TerminalKeyboard  - SSH 세션의 stdin(tty)을 cbreak 모드로 직접 읽는다.
     디스플레이/윈도우가 전혀 필요 없으므로 --headless 든 --livestream 이든 항상 동작한다.
     터미널 자동반복(autorepeat) 덕분에 키를 누르고 있으면 문자가 연속으로 들어온다.
     다만 "뗐다"는 이벤트는 없으므로 값을 누적/유지하는 방식(sticky)으로 다룬다.

  2) CarbKeyboard     - Kit 앱 윈도우의 carb 키보드 이벤트.
     GUI 로 띄웠거나 WebRTC 로 스트리밍 중일 때 동작한다(스트리밍 클라이언트가
     키 입력을 앱 윈도우로 포워딩해 준다). 이쪽은 press/release 가 모두 오므로
     "누르는 동안만 이동" 이 가능하다.

두 경로 모두 같은 KeyboardTeleopState 를 갱신하므로, 켜져 있는 쪽 아무거나 쓰면 된다.
"""

from __future__ import annotations

import atexit
import math
import os
import select
import sys
import time

# cbreak 모드는 POSIX tty 에서만 쓸 수 있다. import 실패 시 터미널 입력만 비활성화된다.
try:
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover - Windows
    _HAS_TERMIOS = False


HELP_TEXT = """
[keyboard]  a/d or Left/Right : 목표 방향 +-{dyaw:.2f} rad (월드 기준 절대 각도)
            space : 목표 방향을 로봇의 현재 진행 방향으로 (지금부터 직진)
            x : 목표 방향 0 (월드 +X, 파쿠르 코스 진행 방향)
            r : 환경 리셋    c : 카메라 전환    ? : 도움말    q or Ctrl-C : 종료
"""


class KeyboardTeleopState:
    """정책에 먹일 사용자 명령값. 입력 경로들이 공유한다.

    target_yaw : 사용자가 정한 **목표 지점 방향**의 월드 기준 절대 각도(rad).
        파쿠르 이벤트가 만드는 target_yaw = atan2(goal_vec_y, goal_vec_x) 와 같은 규약이며,
        즉 "goal point 가 어느 방향에 있는가" 다. 0 이면 월드 +X 방향이다.
        관측의 index 6,7 은 이 값에서 로봇의 현재 진행 각도를 뺀 차이로 계산된다
        (계산은 로봇 자세를 알아야 하므로 demo.py 의 apply_teleop_yaw 가 한다).

        절대 각도이므로 상한을 두지 않고 +-pi 로 감아 준다. 계속 한쪽으로 돌리면
        제자리에서 한 바퀴 돌 수 있다.

    속도는 사용자가 건드리지 않는다. obs index 9 의 command velocity 는 환경의
    command manager 가 정하는 값을 그대로 쓴다.
    """

    def __init__(self, dyaw: float = 0.1):
        self.dyaw = dyaw

        self.target_yaw = 0.0
        # 한 번만 소비되는 일회성 플래그들. 메인 루프가 읽고 지운다.
        self.reset_requested = False
        self.toggle_camera_requested = False
        self.align_requested = False
        self.quit_requested = False
        self.dirty = True
        # 메인 루프가 채워 주는 표시용 값들.
        # sim_fps: 실측 시뮬 속도. 목표는 50Hz 이고, 이보다 한참 낮으면 렌더가 병목이다.
        # delta_yaw: 실제로 obs[6],[7] 에 들어간 값. 조종이 먹고 있는지 눈으로 확인용.
        self.sim_fps = None
        self.delta_yaw = None

    # -- 명령 조작 -------------------------------------------------------
    def add_yaw(self, delta: float):
        self.set_target_yaw(self.target_yaw + delta)

    def set_target_yaw(self, value: float):
        # atan2 규약과 맞추기 위해 항상 (-pi, pi] 로 감는다.
        self.target_yaw = (value + math.pi) % (2 * math.pi) - math.pi
        self.dirty = True

    # -- 공통 키 처리 ----------------------------------------------------
    def handle(self, key: str) -> bool:
        """정규화된 키 이름을 하나 처리한다. 처리했으면 True."""
        if key in ("a", "left"):
            self.add_yaw(self.dyaw)
        elif key in ("d", "right"):
            self.add_yaw(-self.dyaw)
        elif key == "space":
            # 로봇 자세는 여기서 모르므로 메인 루프에 요청만 남긴다.
            self.align_requested = True
        elif key == "x":
            self.set_target_yaw(0.0)
        elif key == "r":
            self.reset_requested = True
        elif key == "c":
            self.toggle_camera_requested = True
        elif key in ("q", "escape"):
            self.quit_requested = True
        elif key in ("?", "/", "h"):
            print(HELP_TEXT.format(dyaw=self.dyaw).replace("\n", "\r\n"), flush=True)
        else:
            return False
        return True

    def status_line(self) -> str:
        delta = f"   delta={self.delta_yaw:+.2f}" if self.delta_yaw is not None else ""
        fps = f"   {self.sim_fps:5.1f} Hz" if self.sim_fps is not None else ""
        return f"[keyboard] goal={self.target_yaw:+.2f} rad{delta}{fps}   (? for help)"


class TerminalKeyboard:
    """SSH 터미널(stdin)에서 키를 논블로킹으로 읽는다.

    setraw 가 아니라 setcbreak 를 쓴다. cbreak 는 ISIG 를 남겨 두므로 Ctrl-C 가
    그대로 SIGINT 로 동작한다(raw 였다면 그냥 0x03 문자로 들어와 버린다).
    """

    def __init__(self, state: KeyboardTeleopState):
        self.state = state
        self._fd = None
        self._saved = None
        self.enabled = False

        if not _HAS_TERMIOS:
            print("[keyboard] termios 가 없어 터미널 입력을 비활성화한다.")
            return
        if not sys.stdin.isatty():
            print("[keyboard] stdin 이 tty 가 아니다(리다이렉트/백그라운드 실행?). 터미널 입력 비활성화.")
            return

        self._fd = sys.stdin.fileno()
        try:
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except termios.error as exc:
            print(f"[keyboard] tty 설정 실패({exc}). 터미널 입력 비활성화.")
            self._saved = None
            return

        self.enabled = True
        # 예외로 죽더라도 터미널이 먹통이 되지 않게 반드시 되돌린다.
        atexit.register(self.close)
        print(HELP_TEXT.format(dyaw=state.dyaw))

    def poll(self):
        """읽을 수 있는 만큼만 읽고 즉시 돌아온다. 시뮬 루프를 막지 않는다."""
        if not self.enabled:
            return
        while select.select([self._fd], [], [], 0)[0]:
            data = os.read(self._fd, 64)
            if not data:
                break
            self._feed(data.decode("utf-8", errors="ignore"))

    def _feed(self, buf: str):
        i = 0
        n = len(buf)
        while i < n:
            ch = buf[i]
            # 방향키는 ESC [ A/B/C/D 로 들어온다.
            if ch == "\x1b" and i + 2 < n and buf[i + 1] == "[":
                arrow = {"A": "up", "B": "down", "C": "right", "D": "left"}.get(buf[i + 2])
                if arrow is not None:
                    self.state.handle(arrow)
                    i += 3
                    continue
            if ch == "\x1b":
                # 단독 ESC. 뒤에 아무것도 안 붙어 있으면 종료로 본다.
                self.state.handle("escape")
            elif ch == " ":
                self.state.handle("space")
            else:
                self.state.handle(ch.lower())
            i += 1

    def close(self):
        if self._saved is not None and self._fd is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._saved = None
        self.enabled = False


class CarbKeyboard:
    """Kit 앱 윈도우의 키보드를 **폴링**해서 읽는다.

    왜 구독이 아니라 폴링인가:
      원래는 subscribe_to_keyboard_events() 로 이벤트를 받았다. 그런데 데스크톱에서
      GUI 로 띄우면 Isaac Sim 창이 포커스를 가져가고, 그 상태에서 실제 키를 눌러도
      우리 콜백이 **한 번도 호출되지 않는다**(XTEST 로 진짜 키 이벤트를 넣어 확인).
      Kit 의 UI/액션 레이어가 먼저 이벤트를 소비해 버리기 때문이다. 특히 스페이스는
      Kit 타임라인 재생/정지에, 방향키는 뷰포트·스테이지 탐색에 이미 묶여 있다.

      get_keyboard_value() 는 이벤트 전파와 무관하게 carb 가 들고 있는 **현재 키 상태**를
      그대로 읽는다. 누가 이벤트를 소비하든 상관없이 값이 보이므로 이쪽이 안정적이다.

    헤드리스(윈도우 없음)면 get_keyboard() 가 None 이거나 예외가 나므로 조용히 꺼진다.
    그때는 TerminalKeyboard 만 쓰면 된다.

    폴링은 레벨(누르고 있는가)만 알려 주므로 눌린 순간을 직접 잡아낸다. 방향키는
    터미널 자동반복과 비슷하게 잠깐 누르고 있으면 연속 입력되도록 하고, 나머지 키는
    누른 순간 한 번만 처리한다(리셋이 연타되면 곤란하다).
    """

    _KEY_MAP = {
        "A": "a",
        "D": "d",
        "LEFT": "left",
        "RIGHT": "right",
        "SPACE": "space",
        "X": "x",
        "R": "r",
        "C": "c",
        "Q": "q",
        "ESCAPE": "escape",
    }
    # 누르고 있으면 반복 입력되는 키. 방향 조절만 해당한다.
    _REPEAT_KEYS = frozenset({"a", "d", "left", "right"})
    _REPEAT_DELAY = 0.35   # 첫 반복까지 기다리는 시간(초)
    _REPEAT_PERIOD = 0.06  # 그 뒤 반복 간격(초)

    def __init__(self, state: KeyboardTeleopState):
        self.state = state
        self.enabled = False
        self._keyboard = None
        self._input = None
        self._codes = {}
        # 키 이름 -> 눌리기 시작한 시각. 안 눌려 있으면 키가 없다.
        self._down_since = {}
        self._next_repeat = {}
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard() if app_window is not None else None
            if keyboard is None:
                print("[keyboard] Kit 앱 윈도우에 키보드가 없다(헤드리스). carb 입력 비활성화.")
                return
            self._input = carb.input.acquire_input_interface()
            if not hasattr(self._input, "get_keyboard_value"):
                print("[keyboard] carb 에 get_keyboard_value 가 없다. carb 입력 비활성화.")
                return
            self._keyboard = keyboard
            self._codes = {
                name: getattr(carb.input.KeyboardInput, carb_name)
                for carb_name, name in self._KEY_MAP.items()
                if hasattr(carb.input.KeyboardInput, carb_name)
            }
            self.enabled = True
            print("[keyboard] carb 키보드 폴링 준비 완료 (Isaac Sim 창에 포커스를 두면 동작).")
        except Exception as exc:  # noqa: BLE001 - 어떤 이유든 실패하면 터미널 입력만 쓰면 된다
            print(f"[keyboard] carb 키보드 초기화 실패({exc}). 터미널 입력만 사용한다.")

    def poll(self):
        """눌린 키를 확인해 상태에 반영한다. 시뮬 루프에서 매 스텝 부른다."""
        if not self.enabled:
            return
        now = time.monotonic()
        for key, code in self._codes.items():
            try:
                pressed = self._input.get_keyboard_value(self._keyboard, code) != 0
            except Exception:  # noqa: BLE001 - 종료 중 카브 인터페이스가 먼저 사라질 수 있다
                return
            if not pressed:
                self._down_since.pop(key, None)
                self._next_repeat.pop(key, None)
                continue
            if key not in self._down_since:
                # 눌린 순간. 한 번 처리한다.
                self._down_since[key] = now
                self._next_repeat[key] = now + self._REPEAT_DELAY
                self.state.handle(key)
            elif key in self._REPEAT_KEYS and now >= self._next_repeat[key]:
                self._next_repeat[key] = now + self._REPEAT_PERIOD
                self.state.handle(key)

    def close(self):
        self.enabled = False
        self._keyboard = None
        self._down_since.clear()
        self._next_repeat.clear()


class KeyboardTeleop:
    """터미널 입력과 carb 입력을 함께 켜 두는 편의 래퍼."""

    def __init__(self, state: KeyboardTeleopState, use_terminal: bool = True, use_carb: bool = True):
        self.state = state
        self.terminal = TerminalKeyboard(state) if use_terminal else None
        self.carb_keyboard = CarbKeyboard(state) if use_carb else None
        self._last_status = ""

    @property
    def any_enabled(self) -> bool:
        return bool((self.terminal and self.terminal.enabled) or (self.carb_keyboard and self.carb_keyboard.enabled))

    def poll(self):
        if self.terminal is not None:
            self.terminal.poll()
        if self.carb_keyboard is not None:
            self.carb_keyboard.poll()
        if self.state.dirty:
            self.state.dirty = False
            line = self.state.status_line()
            if line != self._last_status:
                self._last_status = line
                # cbreak 모드에서는 개행이 \r 을 자동으로 붙이지 않으므로 직접 붙인다.
                sys.stdout.write("\r" + line + "\x1b[K")
                sys.stdout.flush()

    def close(self):
        if self.terminal is not None:
            self.terminal.close()
        if self.carb_keyboard is not None:
            self.carb_keyboard.close()
        sys.stdout.write("\r\n")
        sys.stdout.flush()
