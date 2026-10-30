from __future__ import annotations

import pytest

from northcua.experiments.cases.catalog import CaseCatalog, CaseCatalogLoader
from northcua.experiments.cases.catalog.loader import default_case_catalog_path
from northcua.experiments.cases.contracts import (
    AppProfileId,
    AppProfileSpec,
    CaseSpec,
    DesktopSessionSpec,
    InitRecipeSpec,
    LaunchCommandRecipeArgs,
    OpenBlankAppRecipeArgs,
    OpenDirectoryRecipeArgs,
    OpenFileRecipeArgs,
    OpenUrlRecipeArgs,
    RecipeKind,
    TemplateProfileId,
    TemplateProfileSpec,
    WaitWindowRecipeArgs,
)
from northcua.experiments.cases.recipes import RecipeRenderContext, default_recipe_registry
from northcua.experiments.cases.recipes.shell import launch_background_shell


def test_catalog_loader_reads_bundled_profiles() -> None:
    catalog = CaseCatalogLoader().load()

    assert catalog.template("desktop").alias == "northcua-desktop"
    assert catalog.template("web").alias == "northcua-web"
    assert catalog.apps == {}
    assert "northcua/experiments/cases/catalog/data" in str(default_case_catalog_path())


def test_catalog_accepts_external_profile_ids() -> None:
    catalog = CaseCatalog.from_raw(
        {
            "templates": [{"profile_id": "custom.profile", "alias": "custom-template", "provider": "custom-provider"}],
            "apps": [
                {
                    "app_profile": "custom.app",
                    "template_profile": "custom.profile",
                    "display_name": "Custom App",
                    "launch_binary": "custom",
                    "launch_argv": ["custom", "--safe"],
                }
            ],
        }
    )

    assert catalog.app("custom.app").launch_argv == ["custom", "--safe"]
    assert catalog.template("custom.profile").provider.value == "custom-provider"


def test_catalog_rejects_unknown_app_template_reference() -> None:
    raw = {
        "templates": [{"profile_id": "office-web", "alias": "office"}],
        "apps": [
            {
                "app_profile": "browser.chrome",
                "template_profile": "creative",
                "display_name": "Chrome",
                "launch_binary": "google-chrome",
            }
        ],
    }

    with pytest.raises(ValueError, match="unknown template"):
        CaseCatalog.from_raw(raw)


def test_open_url_recipe_renders_stable_chrome_launch_command() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(task_id="browser", query="Open it", app_profile=AppProfileId.BROWSER_CHROME)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.BROWSER_CHROME),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.OFFICE_WEB),
        remote_workdir="/tmp/northcua-cases/browser",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_URL,
        args=OpenUrlRecipeArgs(app=AppProfileId.BROWSER_CHROME, url="https://example.com"),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    assert "google-chrome" in commands[0].command
    assert "https://example.com" in commands[0].command
    assert commands[0].label == "open-url-browser.chrome"


def test_open_file_recipe_defaults_to_app_window_hint_not_filename() -> None:
    """Opening generated files should wait on app readiness, not brittle file names."""

    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(task_id="browser-file", query="Open local HTML", app_profile=AppProfileId.BROWSER_CHROME)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.BROWSER_CHROME),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.OFFICE_WEB),
        remote_workdir="/tmp/northcua-cases/browser-file",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_FILE,
        args=OpenFileRecipeArgs(
            app=AppProfileId.BROWSER_CHROME,
            path="/home/user/Desktop/NorthCounty-DMV-Research/index.html",
        ),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    assert "index.html" in commands[0].command
    assert "Google Chrome" in commands[0].command
    assert "xdotool search --name index.html" not in commands[0].command


def test_open_file_recipe_converts_calc_csv_to_native_ods_before_launch() -> None:
    """Calc CSV fixtures should follow the native-ODS live-case pattern."""

    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(task_id="calc-csv", query="Open spreadsheet", app_profile=AppProfileId.LIBREOFFICE_CALC)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.LIBREOFFICE_CALC),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.OFFICE_WEB),
        remote_workdir="/tmp/northcua-cases/calc-csv",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_FILE,
        args=OpenFileRecipeArgs(
            app=AppProfileId.LIBREOFFICE_CALC,
            path="/home/user/Desktop/March_Invoice_Line_Items.csv",
            timeout_seconds=60,
        ),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    command = commands[0].command
    assert "--headless --convert-to ods" in command
    assert "/tmp/northcua-cases/calc-csv/March_Invoice_Line_Items.ods" in command
    assert "LibreOffice Calc" in command
    assert "--nofirststartwizard" in command


def test_wait_window_recipe_prefers_window_probe_when_title_is_declared() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(task_id="vscode", query="Wait for app", app_profile=AppProfileId.VS_CODE)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.VS_CODE),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.DEV),
        remote_workdir="/tmp/northcua-cases/vscode",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.WAIT_WINDOW,
        args=WaitWindowRecipeArgs(title_contains="Visual Studio Code"),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    assert "export DISPLAY=${DISPLAY:-:0}" in commands[0].command
    assert "xdotool search --name" in commands[0].command
    assert "case recipe window readiness probe timed out" in commands[0].command
    assert "process_fallback_deadline" not in commands[0].command
    assert "continuing with process fallback" not in commands[0].command


def test_open_file_recipe_requires_window_even_when_process_is_seen() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(task_id="image", query="Open image", app_profile=AppProfileId.IMAGE_VIEWER)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.IMAGE_VIEWER),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.DESKTOP_FULL),
        remote_workdir="/tmp/northcua-cases/image",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_FILE,
        args=OpenFileRecipeArgs(app=AppProfileId.IMAGE_VIEWER, path="/tmp/example.png", wait_window="example.png"),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    assert "xdotool search --name" in commands[0].command
    assert 'kill -0 "$(cat' in commands[0].command
    assert 'now_seconds="$(date +%s)"' in commands[0].command
    assert "now_seconds - start_seconds" in commands[0].command
    assert "then process_seen=1" in commands[0].command
    assert 'process_seen" = "1" ] && exit 0' not in commands[0].command
    assert "process_fallback_deadline" not in commands[0].command
    assert "case recipe window readiness probe timed out after process appeared" in commands[0].command


def test_vscode_launch_allows_process_fallback_for_unstable_window_titles() -> None:
    """VS Code launch recipes keep the proven process fallback from live E2B."""

    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(task_id="vscode-open", query="Open project", app_profile=AppProfileId.VS_CODE)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.VS_CODE),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.DEV),
        remote_workdir="/tmp/northcua-cases/vscode-open",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_DIRECTORY,
        args=OpenDirectoryRecipeArgs(
            path="/home/user/Desktop/project",
            wait_window="Visual Studio Code",
        ),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    assert "process_fallback_deadline" in commands[0].command
    assert 'now_seconds="$(date +%s)"' in commands[0].command
    assert "now_seconds - start_seconds" in commands[0].command
    assert "ps -eo uid=,pid=,etimes=,comm=,args=" in commands[0].command
    assert "$4 !~ /^(sh|bash|dash|zsh|ksh|fish|runuser|sudo|env|setsid|timeout|awk)$/" in commands[0].command
    assert commands[0].command.index("then process_seen=1") < commands[0].command.index('kill -0 "$(cat')


def test_open_file_recipe_uses_app_window_hint_before_file_name() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-osworld-case-catalog.yaml")
    case = CaseSpec(task_id="image", query="Open image", app_profile=AppProfileId.IMAGE_VIEWER)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.IMAGE_VIEWER),
        app_profiles=catalog.apps,
        template=catalog.template("osworld-gnome-full"),
        remote_workdir="/tmp/northcua-cases/image",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_FILE,
        args=OpenFileRecipeArgs(app=AppProfileId.IMAGE_VIEWER, path="/tmp/example.png"),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert "xdotool search --name" in commands[0].command
    assert "Image Viewer" in commands[0].command
    assert "xdotool search --name example.png" not in commands[0].command


def test_open_directory_recipe_uses_app_window_hint_before_directory_name() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-osworld-case-catalog.yaml")
    case = CaseSpec(task_id="files", query="Open folder", app_profile=AppProfileId.FILE_MANAGER)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.FILE_MANAGER),
        app_profiles=catalog.apps,
        template=catalog.template("osworld-gnome-full"),
        remote_workdir="/tmp/northcua-cases/files",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_DIRECTORY,
        args=OpenDirectoryRecipeArgs(app=AppProfileId.FILE_MANAGER, path="/tmp/example-folder"),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert "xdotool search --name org.gnome.Nautilus" in commands[0].command
    assert "xdotool search --name example-folder" not in commands[0].command


def test_osworld_desktop_session_is_typed_and_launch_bound() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-osworld-case-catalog.yaml")
    case = CaseSpec(
        task_id="strict-image",
        query="Open image",
        app_profile=AppProfileId.IMAGE_VIEWER,
        desktop_session=DesktopSessionSpec(target_user="user"),
    )
    template = catalog.template("osworld-gnome-full")
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.IMAGE_VIEWER),
        app_profiles=catalog.apps,
        template=template,
        remote_workdir="/tmp/northcua-cases/strict-image",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_FILE,
        args=OpenFileRecipeArgs(app=AppProfileId.IMAGE_VIEWER, path="/tmp/example.png"),
    )

    command = default_recipe_registry().render(recipe, context)[0].command

    assert "northcua_requested_user=user" in command
    assert "loginctl list-sessions --no-legend" in command
    assert "gnome-shell|gnome-session*" in command
    assert "/tmp/northcua-runtime" not in command
    assert "export DISPLAY=${DISPLAY:-:0}" not in command
    assert 'northcua_session_env_value "$candidate_pid" DBUS_SESSION_BUS_ADDRESS' in command
    assert 'northcua_session_env_value "$candidate_pid" AT_SPI_BUS_ADDRESS' in command
    assert 'northcua_session_env_value "$candidate_pid" DISPLAY' in command
    assert 'northcua_session_env_value "$candidate_pid" XAUTHORITY' in command
    assert 'resolved_runtime=$(readlink -f "$session_runtime"' in command
    assert '"$resolved_runtime" = "/run/user/$launch_uid"' in command
    assert '"$resolved_dbus_path" = "$session_runtime/bus"' in command
    assert '"$session_runtime"/at-spi/*' in command
    assert "desktop session resolution failed" in command
    assert "desktop session diagnostics" in command
    assert "runtime directory has wrong owner" in command
    assert "DBus socket is stale or outside" in command
    assert "AT-SPI bus address is stale" in command
    assert "XAUTHORITY has wrong owner" in command
    assert "open-file-desktop.image_viewer.pid.windows-before" in command
    assert command.index('search --name ".*"') < command.index("setsid eog")
    assert 'grep -Fqx -- "$window_id" "$northcua_window_snapshot"' in command
    assert 'window_pid=$("${northcua_xdotool[@]}" getwindowpid' in command
    assert 'candidate_uid" = "$launch_uid' in command
    assert 'candidate_sid" = "$northcua_launch_pid' in command
    assert 'recorded_launch_pid" = "$northcua_launch_pid' in command
    assert "wmctrl -lx" not in command
    assert "pgrep -af" not in command


def test_strict_vscode_fallback_requires_a_bound_non_wrapper_process() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-osworld-case-catalog.yaml")
    case = CaseSpec(
        task_id="strict-vscode-wrapper",
        query="Open project",
        app_profile=AppProfileId.VS_CODE,
        desktop_session=DesktopSessionSpec(target_user="user"),
    )
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.VS_CODE),
        app_profiles=catalog.apps,
        template=catalog.template("osworld-gnome-full"),
        remote_workdir="/tmp/northcua-cases/strict-vscode-wrapper",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.LAUNCH_COMMAND,
        args=LaunchCommandRecipeArgs(
            app=AppProfileId.VS_CODE,
            command="sleep 30; code /home/user/Desktop/project",
            wait_window="Visual Studio Code",
        ),
    )

    command = default_recipe_registry().render(recipe, context)[0].command

    assert "process_fallback_deadline=$((start_seconds + 8))" in command
    assert 'northcua_process_is_bound "$candidate_pid" || continue' in command
    assert 'candidate_comm=$(cat "/proc/$candidate_pid/comm"' in command
    assert 'case "$candidate_comm" in sh|bash|dash|zsh|ksh|fish' in command
    assert '[[ "$candidate_comm $candidate_args" == *"$northcua_expected_process"* ]]' in command
    assert 'actual_identity" = "$northcua_launch_identity" ]; then return 0' not in command


def test_desktop_session_without_launch_user_discovers_the_live_session() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(
        task_id="strict-root-browser",
        query="Open browser",
        app_profile=AppProfileId.BROWSER_CHROME,
        desktop_session=DesktopSessionSpec(),
    )
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.BROWSER_CHROME),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.OFFICE_WEB),
        remote_workdir="/tmp/northcua-cases/strict-root-browser",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_URL,
        args=OpenUrlRecipeArgs(app=AppProfileId.BROWSER_CHROME, url="https://example.com"),
    )

    command = default_recipe_registry().render(recipe, context)[0].command

    assert "northcua_requested_user=" in command
    assert "northcua_requested_user=user" not in command
    assert "loginctl show-session" in command
    assert 'northcua_launch_user="${northcua_user_record%%:*}"' in command
    assert 'runuser -u "$northcua_launch_user"' in command
    assert "/tmp/northcua-runtime" not in command


def test_strict_calc_conversion_uses_the_resolved_session_before_launch() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-osworld-case-catalog.yaml")
    case = CaseSpec(
        task_id="strict-calc-csv",
        query="Open spreadsheet",
        app_profile=AppProfileId.LIBREOFFICE_CALC,
        desktop_session=DesktopSessionSpec(target_user="user"),
    )
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.LIBREOFFICE_CALC),
        app_profiles=catalog.apps,
        template=catalog.template("osworld-gnome-full"),
        remote_workdir="/tmp/northcua-cases/strict-calc-csv",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_FILE,
        args=OpenFileRecipeArgs(
            app=AppProfileId.LIBREOFFICE_CALC,
            path="/home/user/Desktop/March_Invoice_Line_Items.csv",
            timeout_seconds=60,
        ),
    )

    command = default_recipe_registry().render(recipe, context)[0].command

    assert "--headless --convert-to ods" in command
    assert 'runuser -u "$northcua_launch_user"' in command
    assert 'HOME="$northcua_launch_home"' in command
    assert 'DISPLAY="$launch_display"' in command
    assert 'XAUTHORITY="$launch_xauthority"' in command
    assert 'AT_SPI_BUS_ADDRESS="$launch_at_spi"' in command
    assert "sudo -Hu user" not in command
    assert "chown -R user:user" not in command
    assert "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/1000}" not in command
    assert "DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}" not in command
    assert command.index('export DISPLAY="$launch_display"') < command.index("--headless --convert-to ods")
    assert command.index("--headless --convert-to ods") < command.index('search --name ".*"')


def test_open_blank_recipe_can_launch_catalog_app_as_non_root_user() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(task_id="vlc", query="Open media player", app_profile=AppProfileId.VLC)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.VLC),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.DESKTOP_FULL),
        remote_workdir="/tmp/northcua-cases/vlc",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_BLANK_APP,
        args=OpenBlankAppRecipeArgs(app=AppProfileId.VLC),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    assert "runuser -u user -- env" in commands[0].command
    assert 'session_runtime="/run/user/${launch_uid}"' in commands[0].command
    assert "/tmp/northcua-runtime" not in commands[0].command
    assert "desktop runtime directory has the wrong owner" in commands[0].command
    assert 'DBUS_SESSION_BUS_ADDRESS="unix:path=$session_runtime/bus"' in commands[0].command
    assert 'AT_SPI_BUS_ADDRESS="unix:path=$session_runtime/at-spi/bus"' in commands[0].command
    assert 'XDG_RUNTIME_DIR="$runtime_dir"' in commands[0].command
    assert 'LANG="${LANG:-C.UTF-8}"' in commands[0].command
    assert "bash -c" in commands[0].command
    assert "cd /home/user || exit 1; setsid vlc --no-qt-privacy-ask --no-audio" in commands[0].command
    assert "cd /home/user && setsid vlc --no-qt-privacy-ask --no-audio" not in commands[0].command
    assert "touch /tmp/northcua-cases/vlc/open-blank-media.vlc.out" in commands[0].command
    assert "chown user:user" in commands[0].command
    assert "setsid vlc --no-qt-privacy-ask --no-audio" in commands[0].command
    assert "kill -0" in commands[0].command
    assert "open-blank-media.vlc.pid" in commands[0].command


def test_open_blank_recipe_passes_terminal_working_directory() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(task_id="terminal", query="Inspect file", app_profile=AppProfileId.DEV_TERMINAL)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.DEV_TERMINAL),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.DESKTOP_FULL),
        remote_workdir="/tmp/northcua-cases/terminal",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_BLANK_APP,
        args=OpenBlankAppRecipeArgs(
            app=AppProfileId.DEV_TERMINAL,
            working_directory="/home/user/Desktop/SyntheticComputer/win_computer_000000/drives/D/Project",
        ),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    assert (
        "--working-directory=/home/user/Desktop/SyntheticComputer/win_computer_000000/drives/D/Project"
        in commands[0].command
    )


def test_launch_background_shell_does_not_background_cd_guard() -> None:
    command = "setsid libreoffice --calc file.ods </dev/null >/tmp/out 2>/tmp/err & echo $! > /tmp/pid"

    rendered = launch_background_shell(
        command,
        stdout_path="/tmp/out",
        stderr_path="/tmp/err",
        pid_path="/tmp/pid",
        user="user",
        working_directory="/home/user",
    )

    assert "cd /home/user || exit 1; setsid libreoffice --calc file.ods" in rendered
    assert "cd /home/user && setsid libreoffice --calc file.ods" not in rendered


def test_launch_background_shell_preserves_non_strict_user_prelaunch_once() -> None:
    rendered = launch_background_shell(
        "setsid app </dev/null >/tmp/out 2>/tmp/err & echo $! > /tmp/pid",
        stdout_path="/tmp/out",
        stderr_path="/tmp/err",
        pid_path="/tmp/pid",
        user="user",
        prelaunch_lines=("printf ready >/tmp/prelaunch",),
    )

    assert rendered.count("printf ready >/tmp/prelaunch") == 1
    assert rendered.index("printf ready >/tmp/prelaunch") < rendered.index("setsid app")


def test_launch_background_shell_rejects_empty_passthrough_name() -> None:
    with pytest.raises(ValueError, match="unsafe passthrough env var"):
        launch_background_shell(
            "setsid app </dev/null >/tmp/out 2>/tmp/err & echo $! > /tmp/pid",
            stdout_path="/tmp/out",
            stderr_path="/tmp/err",
            pid_path="/tmp/pid",
            user="user",
            passthrough_env_vars=("",),
        )


def test_open_blank_recipe_requires_window_readiness_when_window_hint_is_stable() -> None:
    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(task_id="calc", query="Open spreadsheet", app_profile=AppProfileId.LIBREOFFICE_CALC)
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.LIBREOFFICE_CALC),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.OFFICE_WEB),
        remote_workdir="/tmp/northcua-cases/calc",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.OPEN_BLANK_APP,
        args=OpenBlankAppRecipeArgs(app=AppProfileId.LIBREOFFICE_CALC),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    assert "open-blank-libreoffice.calc.pid" in commands[0].command
    assert "LibreOffice Calc" in commands[0].command
    assert "process_fallback_deadline" not in commands[0].command
    assert "ps -eo pid=,etimes=,args=" not in commands[0].command


def test_launch_command_recipe_uses_app_readiness_hints() -> None:
    """Arbitrary sandbox launch commands still use typed app readiness probes."""

    catalog = CaseCatalogLoader().load("tests/fixtures/scenarios/catalogs/e2b-case-catalog.yaml")
    case = CaseSpec(
        task_id="chrome-settings", query="Open Chrome privacy settings", app_profile=AppProfileId.BROWSER_CHROME
    )
    context = RecipeRenderContext(
        case=case,
        app=catalog.app(AppProfileId.BROWSER_CHROME),
        app_profiles=catalog.apps,
        template=catalog.template(TemplateProfileId.OFFICE_WEB),
        remote_workdir="/tmp/northcua-cases/chrome-settings",
    )
    recipe = InitRecipeSpec(
        recipe=RecipeKind.LAUNCH_COMMAND,
        args=LaunchCommandRecipeArgs(
            app=AppProfileId.BROWSER_CHROME,
            command="google-chrome --new-window chrome://settings/privacy",
            wait_window="Google Chrome",
        ),
    )

    commands = default_recipe_registry().render(recipe, context)

    assert len(commands) == 1
    assert "google-chrome --new-window chrome://settings/privacy" in commands[0].command
    assert "Google Chrome" in commands[0].command
    assert commands[0].label == "launch-command-browser.chrome"
