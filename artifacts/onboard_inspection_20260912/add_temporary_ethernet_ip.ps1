$ErrorActionPreference = 'Stop'
$resultPath = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\artifacts\onboard_inspection_20260912\network_fix.json'
try {
    $adapter = Get-NetAdapter -InterfaceIndex 21
    if ($adapter.Name -ne 'Ethernet' -or $adapter.InterfaceDescription -notlike 'Killer E3100X*') { throw 'Ethernet adapter identity changed; no changes made.' }
    $selectedAddress = $null
    foreach ($candidateAddress in @('192.168.123.222', '192.168.123.223', '192.168.123.224')) {
        $existing = Get-NetIPAddress -InterfaceIndex 21 -AddressFamily IPv4 | Where-Object { $_.IPAddress -eq $candidateAddress }
        $addedByThisScript = -not $existing
        if ($addedByThisScript) { New-NetIPAddress -InterfaceIndex 21 -IPAddress $candidateAddress -PrefixLength 24 -PolicyStore ActiveStore | Out-Null }
        for ($attempt = 0; $attempt -lt 8; $attempt++) {
            Start-Sleep -Seconds 1
            $address = Get-NetIPAddress -InterfaceIndex 21 -IPAddress $candidateAddress
            if ($address.AddressState -ne 'Tentative') { break }
        }
        if ($address.AddressState -eq 'Preferred') { $selectedAddress = $candidateAddress; break }
        if ($addedByThisScript) { Remove-NetIPAddress -InterfaceIndex 21 -IPAddress $candidateAddress -PolicyStore ActiveStore -Confirm:$false }
    }
    if (-not $selectedAddress) { throw 'No candidate address passed Windows duplicate address detection.' }
    $addresses = Get-NetIPAddress -InterfaceIndex 21 -AddressFamily IPv4 | Select-Object IPAddress,PrefixLength,AddressState
    @{ ok = $true; selected_address = $selectedAddress; addresses = $addresses; timestamp = (Get-Date).ToString('o') } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $resultPath
} catch {
    @{ ok = $false; error = $_.Exception.Message; timestamp = (Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $resultPath
    exit 1
}
