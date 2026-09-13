param([string]$Path='E:\codex-artifacts\sonic23_teleop_resume_20260911\one_step_policy_branch_collection_resume2969_v1\collection\progress.json')
$stream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
try{
    $reader=New-Object IO.StreamReader($stream)
    try{$reader.ReadToEnd()}finally{$reader.Dispose()}
}finally{$stream.Dispose()}
