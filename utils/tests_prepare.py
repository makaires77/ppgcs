import os
import json
import subprocess

class TestEnvironmentPreparer:
    def __init__(self, log_file='preparation_log.txt'):
        self.original_settings = {}
        self.log_file = log_file

    def log(self, message):
        with open(self.log_file, 'a') as file:
            file.write(f"{message}\n")

    def run_command_with_sudo(self, command):
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout
            error = result.stderr

            # Imprimir saídas no terminal
            if output:
                print("Saída:", output)
            if error:
                print("Erro:", error)

            self.log(f"Executado: {command}\nResultado: {output}\nErro: {error}")
            return output
        except subprocess.CalledProcessError as e:
            self.log(f"Erro ao executar o comando: {command}\nErro: {e}")
            print(f"Erro ao executar o comando: {command}\nErro: {e}")
            return ""

    ## Medidas para guardar configurações antes de realizar alterações para testes
    def save_current_settings(self):
        # Salvar as configurações atuais
        self.original_settings['cpu_governor'] = self.get_current_cpu_governor()
        self.original_settings['screensaver_settings'] = self.get_current_screensaver_settings()
        self.original_settings['power_settings'] = self.get_current_power_settings()

        with open('settings_backup.json', 'w') as f:
            json.dump(self.original_settings, f)

    def get_current_cpu_governor(self):
        # Obter a configuração atual do governor da CPU
        result = subprocess.run(["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"], capture_output=True, text=True)
        return result.stdout.strip()

    def get_current_screensaver_settings(self):
        # Obter as configurações atuais do screensaver
        lock_enabled = subprocess.run(["gsettings", "get", "org.gnome.desktop.screensaver", "lock-enabled"], capture_output=True, text=True).stdout.strip()
        idle_activation = subprocess.run(["gsettings", "get", "org.gnome.desktop.screensaver", "idle-activation-enabled"], capture_output=True, text=True).stdout.strip()
        return {'lock_enabled': lock_enabled, 'idle_activation': idle_activation}

    def get_current_power_settings(self):
        # Obter as configurações atuais de energia
        idle_delay = subprocess.run(["gsettings", "get", "org.gnome.desktop.session", "idle-delay"], capture_output=True, text=True).stdout.strip()
        return {'idle_delay': idle_delay}

    def restore_original_settings(self):
        # Restaurar as configurações originais
        if 'cpu_governor' in self.original_settings:
            subprocess.run(["sudo", "cpufreq-set", "-r", "-g", self.original_settings['cpu_governor']], check=True)

        if 'screensaver_settings' in self.original_settings:
            subprocess.run(["gsettings", "set", "org.gnome.desktop.screensaver", "lock-enabled", self.original_settings['screensaver_settings']['lock_enabled']], check=True)
            subprocess.run(["gsettings", "set", "org.gnome.desktop.screensaver", "idle-activation-enabled", self.original_settings['screensaver_settings']['idle_activation']], check=True)

        if 'power_settings' in self.original_settings:
            subprocess.run(["gsettings", "set", "org.gnome.desktop.session", "idle-delay", self.original_settings['power_settings']['idle_delay']], check=True)

        print("Configurações originais restauradas.")

    ## Medidas aplicadas para preparar o ambiente de testes
    def close_unnecessary_applications(self):
        # Esta função depende das aplicações específicas que você deseja fechar
        # Exemplo: Fechando o navegador Firefox
        os.system("pkill firefox")

    def disable_automatic_updates(self):
        # Desativar atualizações automáticas requer privilégios de superusuário
        # e pode variar dependendo da versão do Ubuntu e dos gerenciadores de pacotes.
        # Este é um exemplo usando o comando 'apt'
        self.run_command_with_sudo("systemctl disable --now apt-daily.timer apt-daily-upgrade.timer")

    def configure_power_settings(self):
        # Configura o sistema para desempenho máximo
        self.set_performance_mode()

        # Desativa o screensaver e bloqueio de tela
        self.disable_screensaver_and_lock()

        # Desativa a suspensão e hibernação
        self.disable_suspend_and_hibernation()

    def set_performance_mode(self):
        # Use run_command_with_sudo para executar o comando
        self.run_command_with_sudo("sudo cpufreq-set -r -g performance")

    def disable_screensaver_and_lock(self):
        # Desativa o screensaver
        subprocess.run(["gsettings", "set", "org.gnome.desktop.screensaver", "lock-enabled", "false"], check=True)
        subprocess.run(["gsettings", "set", "org.gnome.desktop.screensaver", "idle-activation-enabled", "false"], check=True)

        # Desativa o bloqueio de tela automático
        subprocess.run(["gsettings", "set", "org.gnome.desktop.session", "idle-delay", "0"], check=True)

    def disable_suspend_and_hibernation(self):
        # Desativa suspensão e hibernação (requer privilégios de superusuário)
        subprocess.run(["sudo", "systemctl", "mask", "sleep.target", "suspend.target", "hibernate.target", "hybrid-sleep.target"], check=True)

    def prepare_test_environment(self):
        self.save_current_settings()
        self.close_unnecessary_applications()
        self.disable_automatic_updates()
        self.configure_power_settings()
        self.log("Ambiente preparado para os testes.")
        print("Ambiente preparado para os testes.")