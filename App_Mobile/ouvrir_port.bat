@echo off
echo Ouverture du port 5000 dans le pare-feu Windows...
netsh advfirewall firewall delete rule name="Flask HAR API" >nul 2>&1
netsh advfirewall firewall add rule name="Flask HAR API" dir=in action=allow protocol=TCP localport=5000
echo.
echo Port 5000 ouvert avec succes !
echo Tu peux fermer cette fenetre.
pause
